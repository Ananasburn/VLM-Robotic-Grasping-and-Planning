import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum

import pinocchio
import mujoco

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)


class PlannerType(Enum):
    """路径规划器类型"""
    RRT_CONNECT = "rrt_connect"
    RL_POLICY = "rl_policy"
    HYBRID = "hybrid"  # 先尝试RL,失败则回退到RRT


class RLPlanner:

    
    def __init__(
        self,
        model_path: str,
        pin_model: Any = None,
        pin_data: Any = None,
        mj_model: Any = None,
        mj_data: Any = None,
        max_steps: int = 500,
        dt: float = 0.02,
    ):
        """
        初始化RL规划器
        
        Args:
            model_path: 训练好的模型路径 (.zip文件)
            pin_model: Pinocchio模型 (用于正向运动学)
            pin_data: Pinocchio数据
            mj_model: MuJoCo模型 (用于碰撞检测)
            mj_data: MuJoCo数据
            max_steps: 最大步数
            dt: 时间步长
        """
        self.model_path = model_path
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.max_steps = max_steps
        self.dt = dt
        
        self.policy = None
        self.n_joints = 6
        self.max_joint_velocity = 1.0
        self.success_threshold = 0.2  # Must match RLPathEnv
        
        # 关节限制
        self.joint_limits_low = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_high = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        try:
            from stable_baselines3 import PPO
            
            if os.path.exists(self.model_path):
                self.policy = PPO.load(self.model_path)
                print(f"✅ RL policy loaded from: {self.model_path}")
            else:
                print(f"⚠️ Model not found: {self.model_path}")
                self.policy = None
        except ImportError:
            print("⚠️ stable_baselines3 not installed. RL planner will not work.")
            self.policy = None
        except Exception as e:
            print(f"❌ Failed to load RL model: {e}")
            self.policy = None
            
    def is_ready(self) -> bool:
        """检查策略是否已加载"""
        return self.policy is not None
        
    def _get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """计算末端执行器位置"""
        if self.pin_model is not None and self.pin_data is not None:
            q_full = np.zeros(self.pin_model.nq)
            q_full[:6] = q
            pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
            pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
            frame_id = self.pin_model.getFrameId("grasp_center")
            return self.pin_data.oMf[frame_id].translation.copy()
        return np.zeros(3)
        
    def _check_collision(self, q: np.ndarray) -> bool:
        """
        检查机器人是否发生碰撞
        
        只检测涉及机器人link的碰撞,忽略:
        - 场景中物体之间的接触 (如Apple与Banana)
        - 机器人底座与地面的接触
        - 其他非机器人相关的接触
        """
        if self.mj_model is None or self.mj_data is None:
            return False
            
        # 设置关节位置并更新仿真
        self.mj_data.qpos[:6] = q
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 如果没有任何接触,则没有碰撞
        if self.mj_data.ncon == 0:
            return False
            
        # 需要忽略的geom (地面、坐标轴、场景物体)
        ignored_geom_names = {
            'floor', 'x-aixs', 'y-aixs', 'z-aixs',  # 地面和坐标轴
            'Apple', 'Banana', 'mocap',              # 场景物体
            'zone_pickup', 'zone_drop',              # 区域标记
            'table1', 'table2', 'simple_table',      # 桌子
            'obstacle_box_1', 'obstacle_sphere_1',   # 障碍物
            'obstacle_sphere_2', 'obstacle_sphere_3',
        }
        
        ignored_geom_ids = set()
        for i in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name in ignored_geom_names:
                ignored_geom_ids.add(i)
                
        # 检查每个接触点
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # 如果任一几何体在忽略列表中,跳过
            if geom1 in ignored_geom_ids or geom2 in ignored_geom_ids:
                continue
                
            # 到这里说明是机器人自碰撞
            # 检查穿透深度
            if contact.dist > -0.01:  # 很小的穿透,忽略
                continue
            # 显著穿透才算碰撞
            return True
                
        return False
        
    def _build_observation(
        self,
        q_current: np.ndarray,
        q_goal: np.ndarray,
        qd_current: np.ndarray,
        prev_action: np.ndarray,
    ) -> np.ndarray:
        """构建观察向量"""
        ee_pos = self._get_ee_position(q_current)
        ee_goal_pos = self._get_ee_position(q_goal)
        
        obs = np.concatenate([
            q_current,       # 6
            qd_current,      # 6
            q_goal,          # 6
            ee_pos,          # 3
            ee_goal_pos,     # 3
            prev_action,     # 6
        ])
        
        return obs.astype(np.float32)
        
    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        validate_collision: bool = True,
        verbose: bool = True,
    ) -> Optional[np.ndarray]:
        """
        使用RL策略生成轨迹
        
        Args:
            q_start: 起始关节配置 (6,)
            q_goal: 目标关节配置 (6,)
            validate_collision: 是否验证碰撞
            verbose: 是否打印信息
            
        Returns:
            轨迹数组 (n_joints, n_steps) 或 None (失败时)
        """
        if not self.is_ready():
            if verbose:
                print("❌ RL policy not loaded")
            return None
            
        if verbose:
            print("\n🤖 RL Planner: Generating trajectory...")
            print(f"   Start: {q_start[:3]}...")
            print(f"   Goal:  {q_goal[:3]}...")
            
        # 初始化状态
        q_current = q_start.copy()
        qd_current = np.zeros(6)
        prev_action = np.zeros(6)
        
        trajectory = [q_start.copy()]
        
        for step in range(self.max_steps):
            # 构建观察
            obs = self._build_observation(q_current, q_goal, qd_current, prev_action)
            
            # 获取动作
            action, _ = self.policy.predict(obs, deterministic=True)
            
            # 缩放动作 - 必须与RLPathEnv.step()中的action_scale一致
            action_scale = 0.1  # 每步最大移动0.1 rad
            scaled_action = action * action_scale
            
            # 更新关节位置
            q_new = q_current + scaled_action
            q_new = np.clip(q_new, self.joint_limits_low, self.joint_limits_high)
            
            # 碰撞检测
            if validate_collision and self._check_collision(q_new):
                if verbose:
                    print(f"⚠️ Collision detected at step {step}")
                return None
                
            # 更新状态
            qd_current = (q_new - q_current) / self.dt
            q_current = q_new
            prev_action = action.copy()
            
            trajectory.append(q_current.copy())
            
            # 检查是否到达目标
            dist_to_goal = np.linalg.norm(q_current - q_goal)
            if dist_to_goal < self.success_threshold:
                if verbose:
                    print(f"✅ Goal reached in {step + 1} steps!")
                break
                
        # 检查最终是否成功
        final_dist = np.linalg.norm(trajectory[-1] - q_goal)
        if final_dist > self.success_threshold:
            if verbose:
                print(f"❌ Failed to reach goal. Final distance: {final_dist:.4f}")
            return None
            
        # 转换为轨迹格式 (n_joints, n_steps)
        trajectory = np.array(trajectory).T
        
        if verbose:
            print(f"   Trajectory length: {trajectory.shape[1]} points")
            
        return trajectory
        
    def plan_with_smoothing(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        smoothing_factor: float = 0.1,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        生成轨迹并进行平滑处理
        
        Args:
            q_start: 起始配置
            q_goal: 目标配置
            smoothing_factor: 平滑因子 (0-1, 越大越平滑)
            **kwargs: 传递给plan()的其他参数
            
        Returns:
            平滑后的轨迹
        """
        trajectory = self.plan(q_start, q_goal, **kwargs)
        
        if trajectory is None:
            return None
            
        # 简单的移动平均平滑
        if smoothing_factor > 0 and trajectory.shape[1] > 5:
            window_size = max(3, int(trajectory.shape[1] * smoothing_factor))
            if window_size % 2 == 0:
                window_size += 1
                
            smoothed = np.zeros_like(trajectory)
            half_window = window_size // 2
            
            for i in range(trajectory.shape[1]):
                start_idx = max(0, i - half_window)
                end_idx = min(trajectory.shape[1], i + half_window + 1)
                smoothed[:, i] = trajectory[:, start_idx:end_idx].mean(axis=1)
                
            # 保持起点和终点不变
            smoothed[:, 0] = q_start
            smoothed[:, -1] = q_goal
            
            return smoothed
            
        return trajectory


def load_rl_planner(
    model_path: str,
    pin_model: Any = None,
    pin_data: Any = None,
    mj_model: Any = None,
    mj_data: Any = None,
) -> RLPlanner:
    """
    加载RL规划器
    
    Args:
        model_path: 模型路径
        pin_model: Pinocchio模型
        pin_data: Pinocchio数据
        mj_model: MuJoCo模型
        mj_data: MuJoCo数据
        
    Returns:
        RLPlanner实例
    """
    return RLPlanner(
        model_path=model_path,
        pin_model=pin_model,
        pin_data=pin_data,
        mj_model=mj_model,
        mj_data=mj_data,
    )


def get_rl_traj(
    env,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    rl_planner: Optional[RLPlanner] = None,
    model_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    使用RL规划器获取轨迹 (兼容现有接口)
    
    Args:
        env: 环境对象 (包含模型信息)
        q_start: 起始配置
        q_goal: 目标配置
        rl_planner: RL规划器实例 (可选)
        model_path: 模型路径 (如果未提供规划器)
        
    Returns:
        轨迹 (n_joints, n_steps) 或 None
    """
    # 确保有规划器
    if rl_planner is None:
        if model_path is None:
            # 使用默认模型路径
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(
                root_dir, 
                "manipulator_grasp/rl_path_planner/models/best_model.zip"
            )
            
        rl_planner = load_rl_planner(
            model_path=model_path,
            pin_model=getattr(env, 'model_roboplan', None),
            pin_data=getattr(env, 'data_roboplan', None),
            mj_model=getattr(env, 'model', None),
            mj_data=getattr(env, 'data', None),
        )
        
    if not rl_planner.is_ready():
        print("❌ RL planner not ready. Please train a model first.")
        return None
        
    # 生成轨迹
    trajectory = rl_planner.plan_with_smoothing(
        q_start[:6],  # 只取前6个关节
        q_goal[:6],
        smoothing_factor=0.1,
    )
    
    if trajectory is not None:
        # 扩展到完整关节空间 (包括夹爪)
        full_trajectory = np.zeros((q_start.shape[0], trajectory.shape[1]))
        full_trajectory[:6, :] = trajectory
        
        # 夹爪保持不变
        for i in range(6, q_start.shape[0]):
            full_trajectory[i, :] = q_start[i]
            
        return full_trajectory
        
    return None


if __name__ == "__main__":
    # 测试代码
    print("Testing RL Planner...")
    
    # 使用默认模型路径
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(root_dir, "manipulator_grasp/rl_path_planner/models/best_model.zip")
    
    planner = RLPlanner(model_path=model_path)
    
    if planner.is_ready():
        q_start = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
        q_goal = np.array([0.5, -0.8, 0.8, -1.5, -1.57, 0.0])
        
        trajectory = planner.plan(q_start, q_goal, validate_collision=False)
        
        if trajectory is not None:
            print(f"Generated trajectory: {trajectory.shape}")
        else:
            print("Failed to generate trajectory")
    else:
        print("Model not found. Please train a model first.")
        print(f"Expected path: {model_path}")
