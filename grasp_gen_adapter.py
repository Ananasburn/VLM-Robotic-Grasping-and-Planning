import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d
import torch
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局单例：GraspGenSampler（延迟加载，避免启动时编译 pointnet2_ops）
# ---------------------------------------------------------------------------
_graspgen_sampler = None

# GraspGen gripper config 路径（默认值，可通过环境变量覆盖）
_GRASPGEN_GRIPPER_CONFIG = os.environ.get(
    "GRASPGEN_GRIPPER_CONFIG",
    os.path.join(ROOT_DIR, "GraspGenModels", "checkpoints", "graspgen_franka_panda.yml"),
)


@dataclass
class GraspCandidate:
    """
    Attributes:
        rotation_matrix: 3×3 旋转矩阵（np.ndarray, float64）
        translation: 3D 平移向量（np.ndarray, float64）
        score: 抓取置信度得分 (float)
        width: 夹爪开合宽度 (float, 米)
        depth: 抓取深度 (float, 米)
    """

    rotation_matrix: np.ndarray
    translation: np.ndarray
    score: float
    width: float = 0.08
    depth: float = 0.02

    def to_open3d_geometry_list(self) -> list:
        """
        生成 Open3D 可视化几何体列表。

        Returns:
            包含 Open3D LineSet 的列表
        """
        # 基于 GraspNet-baseline 的夹爪可视化尺寸
        half_w = self.width / 2.0
        d = self.depth
        # 夹爪线框顶点（局部坐标系）
        vertices = np.array([
            [0, 0, 0],             # 0: 手腕中心
            [0, 0, -d],            # 1: 连接板底端
            [-half_w, 0, -d],      # 2: 左指根
            [-half_w, 0, 0],       # 3: 左指尖
            [half_w, 0, -d],       # 4: 右指根
            [half_w, 0, 0],        # 5: 右指尖
        ], dtype=np.float64)

        # 变换到世界坐标
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_matrix
        transform[:3, 3] = self.translation
        vertices_world = (transform[:3, :3] @ vertices.T).T + transform[:3, 3]

        lines = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5]]
        colors = [[0, 1, 0] for _ in lines]  # 绿色

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return [line_set]


# ---------------------------------------------------------------------------
# GraspGen 模型加载
# ---------------------------------------------------------------------------
def _get_graspgen_sampler():

    global _graspgen_sampler
    if _graspgen_sampler is not None:
        return _graspgen_sampler

    logger.info("[GraspGen] 首次加载 GraspGen 模型...")
    logger.info(f"[GraspGen] Gripper config: {_GRASPGEN_GRIPPER_CONFIG}")

    if not os.path.exists(_GRASPGEN_GRIPPER_CONFIG):
        raise FileNotFoundError(
            f"GraspGen gripper config 文件不存在: {_GRASPGEN_GRIPPER_CONFIG}\n"
            f"请下载 checkpoints: git clone https://huggingface.co/adithyamurali/GraspGenModels\n"
            f"或通过环境变量 GRASPGEN_GRIPPER_CONFIG 指定路径。"
        )

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    grasp_cfg = load_grasp_cfg(_GRASPGEN_GRIPPER_CONFIG)
    _graspgen_sampler = GraspGenSampler(grasp_cfg)
    logger.info("[GraspGen] 模型加载完成。")
    return _graspgen_sampler


# ---------------------------------------------------------------------------
# 数据预处理：从 color/depth/mask 构造点云
# ---------------------------------------------------------------------------
def _build_pointcloud_from_images(
    color_input: "np.ndarray | str",
    depth_input: "np.ndarray | str",
    mask_input: "np.ndarray | str",
) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """
    Args:
        color_input: RGB 图像路径或 (H, W, 3) uint8 数组
        depth_input: 深度图路径或 (H, W) float 数组
        mask_input: 分割掩码路径或 (H, W) 数组

    Returns:
        object_points: (N, 3) float32 物体点云坐标
        object_colors: (N, 3) float32 归一化颜色
        cloud_o3d: Open3D PointCloud 对象
    """
    # 1. 读取 color
    if isinstance(color_input, str):
        color = np.array(Image.open(color_input), dtype=np.float32) / 255.0
    elif isinstance(color_input, np.ndarray):
        color = color_input.astype(np.float32)
        if color.max() > 1.0:
            color /= 255.0
    else:
        raise TypeError(f"color_input 类型无效: {type(color_input)}")

    # 2. 读取 depth
    if isinstance(depth_input, str):
        depth = np.array(Image.open(depth_input))
    elif isinstance(depth_input, np.ndarray):
        depth = depth_input
    else:
        raise TypeError(f"depth_input 类型无效: {type(depth_input)}")

    # 3. 读取 mask
    if isinstance(mask_input, str):
        workspace_mask = np.array(Image.open(mask_input))
    elif isinstance(mask_input, np.ndarray):
        workspace_mask = mask_input
    else:
        raise TypeError(f"mask_input 类型无效: {type(mask_input)}")

    # 4. 相机内参（与 grasp_process.py 一致）
    height, width = color.shape[:2]
    fovy = np.pi / 4
    focal = height / (2.0 * np.tan(fovy / 2.0))
    camera = CameraInfo(width, height, focal, focal, width / 2.0, height / 2.0, 1.0)

    # 5. 深度图 → 有组织点云
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # 6. 使用 mask + depth 门限过滤
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask].astype(np.float32)
    color_masked = color[mask].astype(np.float32)

    # 7. 构造 Open3D 点云
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked)
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked)

    return cloud_masked, color_masked, cloud_o3d


def _convert_grasps_to_candidates(
    grasps: torch.Tensor,
    grasp_conf: torch.Tensor,
    default_width: float = 0.08,
) -> list[GraspCandidate]:
    """
    Args:
        grasps: (M, 4, 4) 齐次变换矩阵 (torch.Tensor)
        grasp_conf: (M,) 置信度分数 (torch.Tensor)
        default_width: 默认夹爪宽度（米）

    Returns:
        按置信度降序排列的 GraspCandidate 列表
    """
    if len(grasps) == 0:
        return []

    grasps_np = grasps.cpu().numpy()
    conf_np = grasp_conf.cpu().numpy()

    candidates = []
    for i in range(len(grasps_np)):
        mat = grasps_np[i]  # (4, 4)
        candidates.append(GraspCandidate(
            rotation_matrix=mat[:3, :3].copy(),
            translation=mat[:3, 3].copy(),
            score=float(conf_np[i]),
            width=default_width,
            depth=0.02,
        ))

    # 按置信度降序排列
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 主入口：GraspGen 推理
# ---------------------------------------------------------------------------
def run_graspgen_inference(
    color_path: "np.ndarray | str",
    depth_path: "np.ndarray | str",
    sam_mask_path: "np.ndarray | str | None" = None,
    target_name: Optional[str] = None,
    num_grasps: int = 200,
    grasp_threshold: float = 0.5,
) -> tuple[list[GraspCandidate], o3d.geometry.PointCloud]:
    """
    Args:
        color_path: RGB 图像路径或 numpy 数组
        depth_path: 深度图路径或 numpy 数组
        sam_mask_path: SAM 分割掩码路径或 numpy 数组
        target_name: 目标物体名称（不为 None 时保存可视化到 {target_name}_gg/）
        num_grasps: 生成的抓取候选数量
        grasp_threshold: 置信度阈值（低于此值的候选被丢弃）

    Returns:
        filtered_gg_list: 按综合得分排序的 GraspCandidate 列表
        cloud_o3d: Open3D 点云对象

    """
    # 1. 构造物体点云
    object_points, object_colors, cloud_o3d = _build_pointcloud_from_images(
        color_path, depth_path, sam_mask_path
    )

    if len(object_points) < 100:
        logger.warning(
            f"[GraspGen] 物体点云过少 ({len(object_points)} 点)，可能导致预测失败。"
        )

    # 2. 加载 GraspGen 推理器
    sampler = _get_graspgen_sampler()

    # 3. 推理（GraspGen 接受 (N, 3) 的物体点云）
    from grasp_gen.grasp_server import GraspGenSampler

    logger.info(f"[GraspGen] 开始推理，输入点云: {object_points.shape}")
    grasps, grasp_conf = GraspGenSampler.run_inference(
        object_points,
        sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
    )

    if len(grasps) == 0:
        logger.warning("[GraspGen] 未检测到任何抓取候选！")
        return [], cloud_o3d

    grasp_conf = grasp_conf.cpu()
    grasps = grasps.cpu()

    logger.info(
        f"[GraspGen] 检测到 {len(grasps)} 个抓取候选，"
        f"置信度范围: [{grasp_conf.min():.3f}, {grasp_conf.max():.3f}]"
    )

    # 4. 转为 GraspCandidate 列表
    all_candidates = _convert_grasps_to_candidates(grasps, grasp_conf)

    # 5. 角度筛选（仅保留接近垂直的抓取）
    vertical = np.array([0, 0, 1])
    angle_threshold = np.deg2rad(45)
    filtered = []
    for candidate in all_candidates:
        approach_dir = candidate.rotation_matrix[:, 0]
        cos_angle = np.clip(np.dot(approach_dir, vertical), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(candidate)

    if len(filtered) == 0:
        logger.warning(
            "[GraspGen] 垂直角度筛选后无候选，使用全部预测。"
        )
        filtered = all_candidates
    else:
        logger.info(
            f"[GraspGen] 角度筛选: {len(filtered)}/{len(all_candidates)} 个候选通过 ±45° 阈值。"
        )

    # 6. 综合得分排序（score × 0.9 + 距离得分 × 0.1）
    points = np.asarray(cloud_o3d.points)
    object_center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    distances = [np.linalg.norm(c.translation - object_center) for c in filtered]
    max_distance = max(distances) if distances else 1.0

    grasp_with_composite_scores = []
    for candidate, dist in zip(filtered, distances):
        distance_score = 1 - (dist / max_distance) if max_distance > 0 else 1.0
        composite_score = candidate.score * 0.9 + distance_score * 0.1
        grasp_with_composite_scores.append((candidate, composite_score))

    grasp_with_composite_scores.sort(key=lambda x: x[1], reverse=True)
    filtered_gg_list = [c for c, _ in grasp_with_composite_scores]

    # 7. 可视化保存
    if target_name is not None:
        _save_graspgen_visualizations(
            cloud_o3d=cloud_o3d,
            all_candidates=all_candidates,
            filtered_gg_list=filtered_gg_list,
            grasp_with_composite_scores=grasp_with_composite_scores,
            target_name=target_name,
        )

    return filtered_gg_list, cloud_o3d



def _save_graspgen_visualizations(
    cloud_o3d: o3d.geometry.PointCloud,
    all_candidates: list[GraspCandidate],
    filtered_gg_list: list[GraspCandidate],
    grasp_with_composite_scores: list[tuple],
    target_name: str,
) -> None:

    # 避免循环导入
    from grasp_process import save_pointcloud_image

    save_dir = os.path.join(ROOT_DIR, "Img_grasping", f"{target_name}_gg")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"[GraspGen VIS] 保存可视化到: {save_dir}")

    # 00: 原始物体点云
    path_raw = os.path.join(save_dir, "00_raw_pointcloud.png")
    save_pointcloud_image([cloud_o3d], path_raw)

    # 01: 全部候选总览
    if len(all_candidates) > 0:
        all_geoms = []
        for c in all_candidates:
            all_geoms.extend(c.to_open3d_geometry_list())
        path_all = os.path.join(save_dir, "01_all_grasps_overview.png")
        save_pointcloud_image([cloud_o3d, *all_geoms], path_all)

    # 02: 每个独立候选
    for idx, (candidate, composite_score) in enumerate(grasp_with_composite_scores):
        geoms = candidate.to_open3d_geometry_list()
        filename = f"02_grasp_{idx:03d}_score{composite_score:.3f}.png"
        path_single = os.path.join(save_dir, filename)
        save_pointcloud_image([cloud_o3d, *geoms], path_single)

    # 03: Top-5 综合排序后的候选
    top_n = min(5, len(filtered_gg_list))
    if top_n > 0:
        top_geoms = []
        for c in filtered_gg_list[:top_n]:
            top_geoms.extend(c.to_open3d_geometry_list())
        path_top = os.path.join(save_dir, "03_top5_composite.png")
        save_pointcloud_image([cloud_o3d, *top_geoms], path_top)

    logger.info(
        f"[GraspGen VIS] 保存完成！共 {2 + len(grasp_with_composite_scores) + 1} 张图片。"
    )
