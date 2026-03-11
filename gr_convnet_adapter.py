import os
import sys
import logging
from typing import Optional
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

from GRConvnet.utils.dataset_processing.grasp import detect_grasps
# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GRCONVNET_DIR = os.path.join(ROOT_DIR, 'GRConvnet')

# 将 GRConvnet 目录加入 sys.path 以便导入其子模块
if GRCONVNET_DIR not in sys.path:
    sys.path.insert(0, GRCONVNET_DIR)

# graspnet-baseline 的子模块路径（与 main_vlm.py 保持一致）
for _sub in ('models', 'dataset', 'utils'):
    _p = os.path.join(ROOT_DIR, 'graspnet-baseline', _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402
from grasp_gen_adapter import GraspCandidate  # noqa: E402

logger = logging.getLogger(__name__)


_DEFAULT_CHECKPOINT = os.path.join(
    GRCONVNET_DIR, 'logs', '260219_0055_training_cornell', 'epoch_48_iou_0.98'
)
_CHECKPOINT_PATH = os.environ.get('GRCONVNET_CHECKPOINT', _DEFAULT_CHECKPOINT)

_grconvnet_model = None


def _get_grconvnet_model() -> torch.nn.Module:

    global _grconvnet_model
    if _grconvnet_model is not None:
        return _grconvnet_model

    if not os.path.exists(_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"GR-ConvNet checkpoint 文件不存在: {_CHECKPOINT_PATH}\n"
            f"请确认 GRConvnet/logs/ 目录下有训练好的模型，或通过环境变量 "
            f"GRCONVNET_CHECKPOINT 指定路径。"
        )

    logger.info(f"[GR-ConvNet] 加载模型: {_CHECKPOINT_PATH}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    _grconvnet_model = torch.load(
        _CHECKPOINT_PATH, weights_only=False, map_location=device
    )
    _grconvnet_model.to(device)
    _grconvnet_model.eval()

    logger.info(f"[GR-ConvNet] 模型加载完成，设备: {device}")
    return _grconvnet_model



def _preprocess_for_grconvnet(
    color_input: "np.ndarray | str",
    depth_input: "np.ndarray | str",
    mask_input: "np.ndarray | str | None" = None,
    output_size: int = 224,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Args:
        color_input: RGB 图像路径或 (H, W, 3) uint8/float 数组（BGR 或 RGB 均可）
        depth_input: 深度图路径或 (H, W) float 数组
        mask_input: SAM 分割掩码路径或 (H, W) 数组，None 时退化为中心裁剪
        output_size: GR-ConvNet 期望的输入尺寸，默认 224

    Returns:
        x: 预处理后的输入张量，形状 [1, 4, output_size, output_size]
        crop_offset: (offset_y, offset_x) 裁剪偏移量，用于坐标映射回原图
    """
    # 1. 读取 RGB
    if isinstance(color_input, str):
        rgb = np.array(Image.open(color_input))
    elif isinstance(color_input, np.ndarray):
        rgb = color_input.copy()
        if len(rgb.shape) == 3 and rgb.shape[2] == 3:
            import cv2
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError(f"color_input 类型无效: {type(color_input)}")

    # 2. 读取深度图
    if isinstance(depth_input, str):
        depth = np.array(Image.open(depth_input)).astype(np.float32)
    elif isinstance(depth_input, np.ndarray):
        depth = depth_input.copy().astype(np.float32)
    else:
        raise TypeError(f"depth_input 类型无效: {type(depth_input)}")
    # 确保深度图是 2D
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    # 3. 读取 mask（可选）
    mask = None
    if mask_input is not None:
        if isinstance(mask_input, str):
            mask = np.array(Image.open(mask_input))
        elif isinstance(mask_input, np.ndarray):
            mask = mask_input.copy()
        # 确保 mask 是 2D bool
        if mask is not None and len(mask.shape) == 3:
            mask = mask[:, :, 0]
        if mask is not None:
            mask = mask > 0

    actual_height, actual_width = rgb.shape[:2]

    # 4. 确定裁剪中心：有 mask 时用 bbox 中心，否则用图像中心
    if mask is not None:
        ys, xs = np.where(mask)
        if len(ys) > 0:
            center_y = int((ys.min() + ys.max()) / 2)
            center_x = int((xs.min() + xs.max()) / 2)
        else:
            logger.warning("[GR-ConvNet] mask 全空，退化为图像中心裁剪")
            center_y = actual_height // 2
            center_x = actual_width // 2
    else:
        center_y = actual_height // 2
        center_x = actual_width // 2

    # 5. 计算裁剪窗口，限制在图像边界内
    half = output_size // 2
    top = center_y - half
    left = center_x - half
    # 边界约束
    top = max(0, min(top, actual_height - output_size))
    left = max(0, min(left, actual_width - output_size))
    bottom = top + output_size
    right = left + output_size

    # 6. 裁剪
    rgb_crop = rgb[top:bottom, left:right].copy()
    depth_crop = depth[top:bottom, left:right].copy()

    # 7. mask 外像素归零（让网络聚焦于目标物体）
    # 7. mask 外像素归零
    # 原因：强制置零导致深度图产生剧烈断层（边缘效应），使热力图聚集在物体边缘。
    #      保留背景有助于网络理解上下文。后处理会负责过滤背景抓取。
    # if mask is not None:
    #     mask_crop = mask[top:bottom, left:right]
    #     rgb_crop[~mask_crop] = 0
    #     depth_crop[~mask_crop] = 0.0

    # 8. 归一化（与 CameraData / Image.normalise 保持一致）
    # RGB: float32 [0,1] 后零均值
    rgb_norm = rgb_crop.astype(np.float32) / 255.0
    rgb_norm -= rgb_norm.mean()
    # Depth: 零均值，clip [-1, 1]
    depth_norm = np.clip(depth_crop - depth_crop.mean(), -1.0, 1.0)

    # 9. 拼接为 [1, 4, H, W] 张量
    # RGB: (H, W, 3) -> (3, H, W)
    rgb_chw = rgb_norm.transpose((2, 0, 1))
    # Depth: (H, W) -> (1, H, W)
    depth_chw = depth_norm[np.newaxis, :, :]
    # 拼接: (4, H, W)
    combined = np.concatenate([depth_chw, rgb_chw], axis=0)
    x = torch.from_numpy(combined[np.newaxis].astype(np.float32))

    return x, (top, left)


# ---------------------------------------------------------------------------
# 2D 像素预测 → 3D 相机坐标系抓取位姿
# ---------------------------------------------------------------------------
def _pixel_grasps_to_3d_poses(
    q_img: np.ndarray,
    ang_img: np.ndarray,
    width_img: np.ndarray,
    depth_full: np.ndarray,
    crop_offset: tuple[int, int],
    image_height: int,
    image_width: int,
    n_grasps: int = 10,
) -> list[GraspCandidate]:
    """
    Args:
        q_img: 抓取质量图，[224, 224]
        ang_img: 抓取角度图（弧度），[224, 224]
        width_img: 抓取宽度图（像素），[224, 224]
        depth_full: 完整深度图（原始分辨率），[H, W]
        crop_offset: (offset_y, offset_x) 裁剪偏移量
        image_height: 原始图像高度
        image_width: 原始图像宽度
        n_grasps: 最大检测抓取数量

    Returns:
        GraspCandidate 列表，按质量分数降序排列
    """


    # 1. 检测 2D 抓取候选
    grasps_2d = detect_grasps(q_img, ang_img, width_img, no_grasps=n_grasps)

    if len(grasps_2d) == 0:
        logger.warning("[GR-ConvNet] 未检测到任何 2D 抓取候选")
        return []

    # 2. 相机内参（与 grasp_process.py 的 get_and_process_data 一致）
    fovy = np.pi / 4
    focal = image_height / (2.0 * np.tan(fovy / 2.0))
    cx = image_width / 2.0
    cy = image_height / 2.0

    offset_y, offset_x = crop_offset

    candidates = []
    for grasp in grasps_2d:
        # grasp.center 是 (row, col) 即 (y_224, x_224)
        y_224, x_224 = grasp.center[0], grasp.center[1]

        # 映射回原图坐标
        u = x_224 + offset_x  # 原图列（水平）
        v = y_224 + offset_y  # 原图行（垂直）

        # 边界检查
        v_clamped = int(np.clip(v, 0, image_height - 1))
        u_clamped = int(np.clip(u, 0, image_width - 1))

        # 3. 获取深度值
        z = float(depth_full[v_clamped, u_clamped])

        # 跳过无效深度
        if z <= 0 or z > 2.0:
            logger.debug(
                f"[GR-ConvNet] 跳过抓取点 ({u}, {v})：深度值无效 z={z:.4f}"
            )
            continue

        # 4. 反投影为 3D 相机坐标
        x_3d = (u - cx) * z / focal
        y_3d = (v - cy) * z / focal
        z_3d = z

        translation = np.array([x_3d, y_3d, z_3d], dtype=np.float64)

        # 5. 构造旋转矩阵（标准相机坐标系约定）
        # Z轴 = 接近方向（从相机指向物体）
        # X轴 = 夹爪闭合方向（连接手指的线）
        # GR-ConvNet 输出的 ang 通常是物体的主轴角度（例如长边角度）
        # 对于二指夹爪，我们需要闭合方向垂直于物体主轴（抓取短边）
        # 因此增加 90 度偏移：ang + pi/2
        ang = float(grasp.angle) + np.pi / 2
        cos_a = np.cos(ang)
        sin_a = np.sin(ang)

        # 旋转矩阵列向量：
        # col0 (X轴 - 夹爪闭合方向): [cos, sin, 0]
        # col1 (Y轴 - 夹爪厚度方向): [-sin, cos, 0]
        # col2 (Z轴 - 接近方向): [0, 0, 1]
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float64)

        # 6. 抓取质量分数
        score = float(q_img[y_224, x_224])

        # 7. 抓取宽度（像素 → 米）
        # GR-ConvNet 的 width_img 已经乘以 150.0（post_process），
        # 单位是像素宽度，需要转换为米
        # 使用深度值和焦距将像素宽度转为世界宽度：w_m = w_px * z / focal
        width_px = float(grasp.length) if grasp.length > 0 else float(width_img[y_224, x_224])
        width_m = width_px * z / focal
        # 限制在合理范围内
        width_m = float(np.clip(width_m, 0.01, 0.15))

        candidates.append(GraspCandidate(
            rotation_matrix=rotation_matrix,
            translation=translation,
            score=score,
            width=width_m,
            depth=0.02,
        ))

    # 按质量分数降序排列
    candidates.sort(key=lambda c: c.score, reverse=True)

    logger.info(
        f"[GR-ConvNet] 从 {len(grasps_2d)} 个 2D 检测中转换得到 "
        f"{len(candidates)} 个有效 3D 抓取候选"
    )
    return candidates


# ---------------------------------------------------------------------------
# 点云构造（与 grasp_gen_adapter 的 _build_pointcloud_from_images 相同逻辑）
# ---------------------------------------------------------------------------
def _build_pointcloud(
    color_input: "np.ndarray | str",
    depth_input: "np.ndarray | str",
    mask_input: "np.ndarray | str",
) -> o3d.geometry.PointCloud:
    """
    Args:
        color_input: RGB 图像路径或 (H, W, 3) uint8 数组
        depth_input: 深度图路径或 (H, W) float 数组
        mask_input: 分割掩码路径或 (H, W) 数组

    Returns:
        cloud_o3d: Open3D PointCloud 对象
    """
    # 读取 color
    if isinstance(color_input, str):
        color = np.array(Image.open(color_input), dtype=np.float32) / 255.0
    elif isinstance(color_input, np.ndarray):
        color = color_input.astype(np.float32)
        if color.max() > 1.0:
            color /= 255.0
    else:
        raise TypeError(f"color_input 类型无效: {type(color_input)}")

    # 读取 depth
    if isinstance(depth_input, str):
        depth = np.array(Image.open(depth_input))
    elif isinstance(depth_input, np.ndarray):
        depth = depth_input
    else:
        raise TypeError(f"depth_input 类型无效: {type(depth_input)}")

    # 读取 mask
    if mask_input is None:
        # 无 mask 时使用全深度图 mask（深度 < 2.0 的区域）
        workspace_mask = (depth < 2.0).astype(np.uint8) * 255
    elif isinstance(mask_input, str):
        workspace_mask = np.array(Image.open(mask_input))
    elif isinstance(mask_input, np.ndarray):
        workspace_mask = mask_input
    else:
        raise TypeError(f"mask_input 类型无效: {type(mask_input)}")

    # 相机内参（与 grasp_process.py 一致）
    height, width = color.shape[:2]
    fovy = np.pi / 4
    focal = height / (2.0 * np.tan(fovy / 2.0))
    camera = CameraInfo(width, height, focal, focal, width / 2.0, height / 2.0, 1.0)

    # 深度图 → 有组织点云
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # mask + depth 门限过滤
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask].astype(np.float32)
    color_masked = color[mask].astype(np.float32)

    # 构造 Open3D 点云
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked)
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked)

    return cloud_o3d


# ---------------------------------------------------------------------------
# 主入口：GR-ConvNet 推理
# ---------------------------------------------------------------------------
def run_grconvnet_inference(
    color_path: "np.ndarray | str",
    depth_path: "np.ndarray | str",
    sam_mask_path: "np.ndarray | str | None" = None,
    target_name: Optional[str] = None,
    n_grasps: int = 10,
) -> tuple[list[GraspCandidate], o3d.geometry.PointCloud]:
    """
    Args:
        color_path: RGB 图像路径或 numpy 数组（BGR uint8）
        depth_path: 深度图路径或 numpy 数组（float，单位：米）
        sam_mask_path: SAM 分割掩码路径或 numpy 数组
        target_name: 目标物体名称（不为 None 时保存可视化到 {target_name}_gg/）
        n_grasps: 最大检测抓取数量

    Returns:
        filtered_gg_list: 按综合得分排序的 GraspCandidate 列表
        cloud_o3d: Open3D 点云对象

    Raises:
        FileNotFoundError: 如果 GR-ConvNet checkpoint 不存在
    """
    # 0. 读取原始深度图（用于反投影）
    if isinstance(depth_path, str):
        depth_full = np.array(Image.open(depth_path))
    elif isinstance(depth_path, np.ndarray):
        depth_full = depth_path.copy()
    else:
        raise TypeError(f"depth_path 类型无效: {type(depth_path)}")

    # 获取原图尺寸
    if isinstance(color_path, str):
        color_for_size = np.array(Image.open(color_path))
    elif isinstance(color_path, np.ndarray):
        color_for_size = color_path
    else:
        raise TypeError(f"color_path 类型无效: {type(color_path)}")
    image_height, image_width = color_for_size.shape[:2]

    # 1. 预处理为 GR-ConvNet 输入（传入 mask 实现目标区域裁剪）
    logger.info("[GR-ConvNet] 预处理图像...")
    x, crop_offset = _preprocess_for_grconvnet(
        color_path, depth_path, mask_input=sam_mask_path
    )

    # 2. 加载模型并推理
    model = _get_grconvnet_model()
    device = next(model.parameters()).device

    logger.info("[GR-ConvNet] 执行前向推理...")
    with torch.no_grad():
        xc = x.to(device)
        pred = model.predict(xc)

    # 3. 后处理
    from inference.post_process import post_process_output
    q_img, ang_img, width_img = post_process_output(
        pred['pos'], pred['cos'], pred['sin'], pred['width']
    )

    # 释放 GPU 内存
    del pred, xc
    torch.cuda.empty_cache()

    # 3.5 用 SAM mask 遮蔽 q_img（仅保留物体区域的预测峰值）
    #     虽然输入已做了 mask 裁剪，但网络输出仍可能在 mask 外有次级峰值，
    #     这里强制将 mask 外的质量置零，避免 detect_grasps 检测到伪峰值。
    #     对 mask 做适度膨胀，确保小物体（如 cube）不会因覆盖像素过少
    #     而导致 peak_local_max 找不到峰值。
    if sam_mask_path is not None:
        if isinstance(sam_mask_path, str):
            _mask_full = np.array(Image.open(sam_mask_path))
        elif isinstance(sam_mask_path, np.ndarray):
            _mask_full = sam_mask_path.copy()
        else:
            _mask_full = None

        if _mask_full is not None:
            if len(_mask_full.shape) == 3:
                _mask_full = _mask_full[:, :, 0]
            _mask_bool = _mask_full > 0
            # 裁剪到与预处理相同的 224×224 区域
            top, left = crop_offset
            output_size = q_img.shape[0]  # 224
            _mask_crop = _mask_bool[top:top + output_size, left:left + output_size]
            # 膨胀 mask（确保小物体边缘有足够的有效区域）
            
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (25, 25)
            )
            _mask_crop = cv2.dilate(
                _mask_crop.astype(np.uint8), dilate_kernel, iterations=1
            ).astype(bool)
            # 遮蔽 q_img：mask 外质量置零
            q_img = q_img * _mask_crop.astype(q_img.dtype)
            logger.info(
                f"[GR-ConvNet] mask 遮蔽 q_img（含膨胀）：有效像素 "
                f"{_mask_crop.sum()}/{output_size**2}"
            )

    # 4. 2D → 3D 转换
    all_candidates = _pixel_grasps_to_3d_poses(
        q_img, ang_img, width_img,
        depth_full, crop_offset,
        image_height, image_width,
        n_grasps=n_grasps,
    )

    if len(all_candidates) == 0:
        logger.warning("[GR-ConvNet] 未检测到任何有效抓取候选！")
        # 仍然构造点云返回
        cloud_o3d = _build_pointcloud(color_path, depth_path, sam_mask_path)
        return [], cloud_o3d

    # 5. 角度筛选（仅保留接近垂直的抓取 — 在相机坐标系中）
    vertical = np.array([0, 0, 1])  # 相机 Z 轴方向
    angle_threshold = np.deg2rad(45)
    filtered = []
    for candidate in all_candidates:
        approach_dir = candidate.rotation_matrix[:, 2]  # Z 轴是接近方向（标准约定）
        cos_angle = np.clip(np.dot(approach_dir, vertical), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(candidate)

    if len(filtered) == 0:
        logger.warning(
            "[GR-ConvNet] 垂直角度筛选后无候选，使用全部预测。"
        )
        filtered = all_candidates
    else:
        logger.info(
            f"[GR-ConvNet] 角度筛选: {len(filtered)}/{len(all_candidates)} "
            f"个候选通过 ±45° 阈值。"
        )

    # 6. 构造物体点云
    cloud_o3d = _build_pointcloud(color_path, depth_path, sam_mask_path)

    # 7. 综合得分排序（score × 0.9 + 距离得分 × 0.1）
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

    logger.info(
        f"[GR-ConvNet] 最终返回 {len(filtered_gg_list)} 个抓取候选"
    )

    # 8. 可视化保存
    if target_name is not None:
        _save_grconvnet_visualizations(
            cloud_o3d=cloud_o3d,
            q_img=q_img,
            ang_img=ang_img,
            all_candidates=all_candidates,
            filtered_gg_list=filtered_gg_list,
            grasp_with_composite_scores=grasp_with_composite_scores,
            target_name=target_name,
        )

    return filtered_gg_list, cloud_o3d


# ---------------------------------------------------------------------------
# 可视化保存
# ---------------------------------------------------------------------------
def _save_grconvnet_visualizations(
    cloud_o3d: o3d.geometry.PointCloud,
    q_img: np.ndarray,
    ang_img: np.ndarray,
    all_candidates: list[GraspCandidate],
    filtered_gg_list: list[GraspCandidate],
    grasp_with_composite_scores: list[tuple],
    target_name: str,
) -> None:
    """
    Args:
        cloud_o3d: 物体点云
        q_img: 抓取质量图 [224, 224]
        ang_img: 抓取角度图 [224, 224]
        all_candidates: 所有抓取候选（未经角度筛选）
        filtered_gg_list: 综合评分排序后的最终抓取候选列表
        grasp_with_composite_scores: (GraspCandidate, composite_score) 元组列表
        target_name: 目标物体名称
    """
    from datetime import datetime
    from grasp_process import save_pointcloud_image

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(ROOT_DIR, "Img_grasping", f"{target_name}_gg", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"[GR-ConvNet VIS] 保存可视化到: {save_dir}")
    print(f"\n{'='*60}")
    print(f"[GR-ConvNet VIS] 保存可视化到: {save_dir}")
    print(f"{'='*60}")

    # 00: 原始物体点云
    path_raw = os.path.join(save_dir, "00_raw_pointcloud.png")
    save_pointcloud_image([cloud_o3d], path_raw)
    print(f"[SAVE] 原始点云 → {path_raw}")

    # 00b: 质量图热力图（2D 预测可视化）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(q_img, cmap='jet')
        axes[0].set_title('Quality Map (q_img)')
        axes[0].axis('off')
        axes[1].imshow(ang_img, cmap='hsv')
        axes[1].set_title('Angle Map (ang_img)')
        axes[1].axis('off')
        plt.tight_layout()
        path_heatmap = os.path.join(save_dir, "00b_prediction_heatmaps.png")
        plt.savefig(path_heatmap, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[SAVE] 预测热力图 → {path_heatmap}")
    except Exception as e:
        logger.warning(f"[GR-ConvNet VIS] 无法保存热力图: {e}")

    # 01: 全部候选总览
    if len(all_candidates) > 0:
        all_geoms = []
        for c in all_candidates:
            all_geoms.extend(c.to_open3d_geometry_list())
        path_all = os.path.join(save_dir, "01_all_grasps_overview.png")
        save_pointcloud_image([cloud_o3d, *all_geoms], path_all)
        print(f"[SAVE] 全部候选总览 ({len(all_candidates)}个) → {path_all}")

    # 02: 每个独立候选
    for idx, (candidate, composite_score) in enumerate(grasp_with_composite_scores[:5]):
        geoms = candidate.to_open3d_geometry_list()
        filename = f"02_grasp_{idx:03d}_score{composite_score:.3f}.png"
        path_single = os.path.join(save_dir, filename)
        save_pointcloud_image([cloud_o3d, *geoms], path_single)
        print(
            f"[SAVE] 候选 #{idx:03d} "
            f"(综合得分={composite_score:.3f}, 原始得分={candidate.score:.3f}) "
            f"→ {path_single}"
        )

    # 03: Top-5 综合排序后的候选
    top_n = min(5, len(filtered_gg_list))
    if top_n > 0:
        top_geoms = []
        for c in filtered_gg_list[:top_n]:
            top_geoms.extend(c.to_open3d_geometry_list())
        path_top = os.path.join(save_dir, "03_top5_composite.png")
        save_pointcloud_image([cloud_o3d, *top_geoms], path_top)
        print(f"[SAVE] Top-{top_n} 综合排序候选 → {path_top}")

    total_images = 2 + min(5, len(grasp_with_composite_scores)) + 1
    print(f"\n[GR-ConvNet VIS] 保存完成！共 {total_images} 张图片到 {save_dir}")
    print(f"{'='*60}\n")
