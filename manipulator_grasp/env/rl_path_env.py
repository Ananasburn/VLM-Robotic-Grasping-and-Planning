"""
Gymnasium-compatible RL Environment for Robot Arm Path Planning
基于MuJoCo的强化学习路径规划环境

This environment trains a policy to move the robot arm from start to goal configurations
while avoiding collisions, moving smoothly, and completing as fast as possible.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import pinocchio


class RLPathEnv(gym.Env):
    """
    强化学习路径规划环境
    
    观察空间:
        - 当前关节位置 (6,)
        - 当前关节速度 (6,)
        - 目标关节位置 (6,)
        - 末端执行器位置 (3,)
        - 目标末端执行器位置 (3,)
        - 上一步动作 (6,)
    
    动作空间:
        - 关节速度命令 (6,) 归一化到 [-1, 1]
    
    奖励设计:
        - 接近目标奖励 (dense)
        - 碰撞惩罚
        - 平滑性奖励 (惩罚急剧变化)
        - 成功到达奖励
        - 时间惩罚 (鼓励快速完成)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        reward_config: Optional[Dict] = None,
        randomize_obstacles: bool = False,
        curriculum_level: int = 0,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.randomize_obstacles = randomize_obstacles
        self.curriculum_level = curriculum_level
        
        # 奖励权重配置 - 调整后的参数,避免过大的碰撞惩罚导致训练不稳定
        self.reward_config = reward_config or {
            "distance_weight": 5.0,           # 距离奖励权重 (增加以提供更强的引导)
            "collision_penalty": -10.0,       # 碰撞惩罚 (降低以避免不稳定)
            "smoothness_weight": 0.05,        # 平滑性奖励权重
            "success_bonus": 50.0,            # 成功奖励
            "time_penalty": -0.01,            # 时间惩罚
            "joint_limit_penalty": -5.0,      # 关节限制惩罚
            "velocity_penalty": -0.005,       # 速度惩罚(防止过快运动)
        }
        
        # 机器人参数
        self.n_joints = 6  # UR3e有6个关节
        self.max_joint_velocity = 1.0  # rad/s
        self.success_threshold = 0.2  # 到达目标的阈值 (rad) - 增加以使学习更容易
        
        # 关节限制 (UR3e)
        self.joint_limits_low = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_high = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # MuJoCo模型
        self._load_mujoco_model()
        
        # Pinocchio模型用于正向运动学
        self._load_pinocchio_model()
        
        # 定义观察和动作空间 (增强版: 18维)
        obs_dim = 12 + 6  # q_current(6) + q_goal(6) + ee_pos(3) + ee_goal_pos(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        
        # 典型配置 (抓取区域、放置区域、复位位置)
        self.preset_configs = {
            "home": np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]),
            "grasp_approach": np.array([0.5, -0.8, 0.8, -1.5, -1.57, 0.0]),
            "place": np.array([-0.3, -1.0, 1.2, -1.8, -1.57, 0.0]),
        }
        
        # 状态变量
        self.current_step = 0
        self.q_start = None
        self.q_goal = None
        self.prev_action = np.zeros(self.n_joints)
        self.prev_q = None
        
        # 渲染相关
        self.viewer = None
        self.renderer = None
        
    def _load_mujoco_model(self):
        """加载MuJoCo模型"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scene_path = os.path.join(base_dir, 'assets', 'scenes', 'scene.xml')
        
        if not os.path.exists(scene_path):
            # 使用简化的模型进行训练
            scene_path = os.path.join(base_dir, 'assets', 'scenes', 'scene.xml')
            
        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sim_hz = 500
        
    def _load_pinocchio_model(self):
        """加载Pinocchio模型用于运动学计算"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(base_dir, 'robot_description', 'urdf', 'ur3e_ag95.urdf')
        
        if os.path.exists(urdf_path):
            package_dirs = os.path.dirname(urdf_path)
            self.pin_model = pinocchio.buildModelFromUrdf(urdf_path)
            self.pin_data = self.pin_model.createData()
            self.ee_frame_id = self.pin_model.getFrameId("grasp_center")
        else:
            self.pin_model = None
            self.pin_data = None
            self.ee_frame_id = None
            
    def _get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """计算末端执行器位置"""
        if self.pin_model is not None:
            # 使用Pinocchio计算正向运动学
            q_full = np.zeros(self.pin_model.nq)
            q_full[:6] = q
            pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
            pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
            return self.pin_data.oMf[self.ee_frame_id].translation.copy()
        else:
            # 简化计算 (仅用于测试)
            return np.zeros(3)
            
    def _check_collision(self) -> bool:
        """
        检查机器人是否发生碰撞
        
        只检测涉及机器人link的碰撞,忽略:
        - 场景中物体之间的接触 (如Apple与Banana)
        - 机器人底座与地面的接触
        - 其他非机器人相关的接触
        """
        # 如果没有任何接触,则没有碰撞
        if self.mj_data.ncon == 0:
            return False
            
        # 定义机器人相关的几何体ID范围
        # 根据调试输出,机器人的geom id是4-67 (None名称的都是机器人link)
        # 但不包括 floor(0), x-axis(1), y-axis(2), z-axis(3)
        # 以及场景物体 (>=68)
        
        # 更可靠的方法: 检查geom名称是否包含link相关关键词
        # 或者geom的body属于机器人
        
        # 需要忽略的geom (地面、坐标轴、场景物体)
        ignored_geom_names = {
            'floor', 'x-aixs', 'y-aixs', 'z-aixs',  # 地面和坐标轴
            'Apple', 'Banana', 'mocap',              # 场景物体
            'zone_pickup', 'zone_drop',              # 区域标记
            'table1', 'table2', 'simple_table',      # 桌子
            'obstacle_box_1', 'obstacle_sphere_1',   # 障碍物 (也忽略,因为RL训练时不需要)
            'obstacle_sphere_2', 'obstacle_sphere_3', # 更多障碍物
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
            
            # 如果两个几何体都在忽略列表中,跳过 (如Apple与Banana)
            if geom1 in ignored_geom_ids and geom2 in ignored_geom_ids:
                continue
                
            # 如果只有一个在忽略列表中,另一个是机器人link
            # 这可能是机器人与场景物体的接触
            # 为了训练稳定性,我们暂时也忽略这些
            if geom1 in ignored_geom_ids or geom2 in ignored_geom_ids:
                continue
                
            # 到这里说明是机器人自碰撞 (两个都是机器人link)
            # 检查是否是相邻link的正常接触
            geom1_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
            geom2_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
            
            # 忽略相邻link之间的接触 (这是正常的)
            # 检查是否都是无名称的geom (机器人link通常没有名称)
            if geom1_name == "" and geom2_name == "":
                # 检查穿透深度
                if contact.dist > -0.01:  # 很小的穿透,忽略
                    continue
                # 显著穿透才算碰撞
                return True
                
        return False
        

    def _check_joint_limits(self, q: np.ndarray) -> bool:
        """检查是否超出关节限制"""
        return np.any(q < self.joint_limits_low) or np.any(q > self.joint_limits_high)
        
    def _compute_reward(
        self, 
        q: np.ndarray, 
        action: np.ndarray,
        collision: bool,
        reached_goal: bool
    ) -> Tuple[float, Dict]:
        """
        计算奖励（改进版，基于参考实现）
        
        包含:
        1. 非线性距离奖励 - 越接近目标奖励越高
        2. 直线性奖励 - 鼓励沿直线运动
        3. 偏离惩罚 - 惩罚绕路行为
        4. 碰撞惩罚
        5. 平滑性惩罚
        6. 成功奖励
        """
        info = {}
        reward = 0.0
        
        # 1. 非线性距离奖励（参考实现的关键改进）
        dist_to_goal = np.linalg.norm(q - self.q_goal)
        
        if dist_to_goal < self.success_threshold:
            distance_reward = 100.0
        elif dist_to_goal < 2 * self.success_threshold:
            distance_reward = 50.0
        elif dist_to_goal < 3 * self.success_threshold:
            distance_reward = 10.0
        else:
            distance_reward = 5.0 / (1.0 + dist_to_goal)
        
        reward += distance_reward
        info["distance_reward"] = distance_reward
        info["dist_to_goal"] = dist_to_goal
        
        # 2. 直线性奖励 (稍微降低权重，依赖VecNormalize处理数值范围)
        linearity_reward = 0.0
        deviation_penalty = 0.0
        
        if self.q_start is not None:
            start_to_goal = self.q_goal - self.q_start
            start_to_goal_norm = np.linalg.norm(start_to_goal)
            
            if start_to_goal_norm >= 1e-6:
                # 计算当前位置到起点的向量
                start_to_current = q - self.q_start
                
                # 计算投影比例（限制在0~1）
                projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm ** 2)
                projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
                
                # 计算投影点和偏离距离
                projected_point = self.q_start + projection_ratio * start_to_goal
                linearity_error = np.linalg.norm(q - projected_point)
                
                # 直线接近奖励 (降低系数 2.0 -> 1.0)
                linearity_reward = 1.0 / (1.0 + linearity_error)
                
                # 偏离惩罚
                if not hasattr(self, 'min_linearity_error'):
                    self.min_linearity_error = np.inf
                
                if linearity_error < self.min_linearity_error:
                    self.min_linearity_error = linearity_error
                else:
                    deviation_penalty = -0.5 * (linearity_error - self.min_linearity_error)
        
        reward += linearity_reward + deviation_penalty
        info["linearity_reward"] = linearity_reward
        info["deviation_penalty"] = deviation_penalty
        
        # 3. 碰撞惩罚
        if collision:
            reward += self.reward_config["collision_penalty"]
            info["collision"] = True
        else:
            info["collision"] = False
            
        # 4. 平滑性惩罚
        action_diff = np.linalg.norm(action - self.prev_action)
        smoothness_penalty = -action_diff * self.reward_config["smoothness_weight"]
        reward += smoothness_penalty
        info["smoothness_penalty"] = smoothness_penalty
        
        # 5. 成功奖励
        if reached_goal:
            reward += self.reward_config["success_bonus"]
            info["success"] = True
        else:
            info["success"] = False
            
        # 6. 时间惩罚
        reward += self.reward_config["time_penalty"]
        
        # 7. 关节限制惩罚
        if self._check_joint_limits(q):
            reward += self.reward_config["joint_limit_penalty"]
            info["joint_limit_violation"] = True
        else:
            info["joint_limit_violation"] = False
        
        return reward, info
        
    def _get_observation(self) -> np.ndarray:
        """
        构建观察向量（增强版）
        
        包含:
        - 当前关节位置 (6维)
        - 目标关节位置 (6维)
        - 当前末端位置 (3维)
        - 目标末端位置 (3维)
        
        总共18维，增加空间信息帮助网络更快理解几何关系
        """
        q_current = self.mj_data.qpos[:6].copy()
        q_goal = self.q_goal.copy()
        
        # 计算末端位置
        ee_pos = self._get_ee_position(q_current)
        ee_goal_pos = self._get_ee_position(q_goal)
        
        obs = np.concatenate([q_current, q_goal, ee_pos, ee_goal_pos])
        return obs.astype(np.float32)
        
    def _sample_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        """采样起始和目标配置"""
        if self.curriculum_level == 0:
            # Level 0: 固定简单任务
            q_start = self.preset_configs["home"].copy()
            q_goal = self.preset_configs["grasp_approach"].copy()
        elif self.curriculum_level == 1:
            # Level 1: 在预设点之间随机
            keys = list(self.preset_configs.keys())
            start_key, goal_key = np.random.choice(keys, 2, replace=False)
            q_start = self.preset_configs[start_key].copy()
            q_goal = self.preset_configs[goal_key].copy()
        else:
            # Level 2+: 完全随机 (在可达空间内)
            q_start = np.random.uniform(
                self.joint_limits_low * 0.5,
                self.joint_limits_high * 0.5
            )
            q_goal = np.random.uniform(
                self.joint_limits_low * 0.5,
                self.joint_limits_high * 0.5
            )
            
        return q_start, q_goal
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 解析选项
        if options is not None:
            q_start = options.get("q_start", None)
            q_goal = options.get("q_goal", None)
            self.curriculum_level = options.get("curriculum_level", self.curriculum_level)
        else:
            q_start = None
            q_goal = None
            
        # 采样或使用指定的起始/目标
        if q_start is None or q_goal is None:
            self.q_start, self.q_goal = self._sample_start_goal()
        else:
            self.q_start = np.array(q_start)
            self.q_goal = np.array(q_goal)
            
        # 重置MuJoCo状态
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:6] = self.q_start.copy()
        self.mj_data.ctrl[:6] = self.q_start.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 重置状态变量
        self.current_step = 0
        self.prev_action = np.zeros(self.n_joints)
        self.prev_q = self.q_start.copy()
        
        # 随机化障碍物 (如果启用)
        if self.randomize_obstacles:
            self._randomize_obstacles()
            
        obs = self._get_observation()
        info = {
            "q_start": self.q_start.copy(),
            "q_goal": self.q_goal.copy(),
            "curriculum_level": self.curriculum_level,
        }
        
        return obs, info
        
    def _randomize_obstacles(self):
        """随机化障碍物位置 (用于Domain Randomization)"""
        # 在实际实现中，这里可以修改MuJoCo模型中的障碍物位置
        # 目前作为占位符
        pass
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作
        
        动作解释: action ∈ [-1, 1]^6 表示每个关节的相对位置变化
        action = 1.0 表示该关节增加 action_scale rad
        """
        self.current_step += 1
        
        # 动作缩放 - 每步最大移动量
        # 增加缩放因子使学习更容易
        action_scale = 0.1  # 每步最大移动0.1 rad (约5.7度)
        scaled_action = action * action_scale
        
        # 当前关节位置
        q_current = self.mj_data.qpos[:6].copy()
        
        # 计算目标关节位置
        q_target = q_current + scaled_action
        
        # 裁剪到关节限制内
        q_target = np.clip(q_target, self.joint_limits_low, self.joint_limits_high)
        
        # 设置MuJoCo控制目标
        self.mj_data.ctrl[:6] = q_target
        
        # 执行多步模拟使机器人移动到目标位置
        # 这是因为MuJoCo使用位置控制,需要多步才能到达目标
        n_substeps = 10
        for _ in range(n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)
        
        # 读取实际关节位置
        q_actual = self.mj_data.qpos[:6].copy()
        
        # 检查终止条件
        collision = self._check_collision()
        dist_to_goal = np.linalg.norm(q_actual - self.q_goal)
        reached_goal = dist_to_goal < self.success_threshold
        timeout = self.current_step >= self.max_steps
        
        # 计算奖励
        reward, info = self._compute_reward(q_actual, action, collision, reached_goal)
        
        # 确定终止条件
        terminated = reached_goal or collision
        truncated = timeout
        
        # 更新状态
        self.prev_action = action.copy()
        self.prev_q = q_actual.copy()
        
        # 添加额外信息
        info["step"] = self.current_step
        info["reached_goal"] = reached_goal
        info["timeout"] = timeout
        info["q_actual"] = q_actual.copy()
        info["q_target"] = q_target.copy()
        
        obs = self._get_observation()

        
        return obs, reward, terminated, truncated, info

    def set_curriculum_level(self, level: int):
        """设置课程学习难度级别"""
        self.curriculum_level = level
        
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
            self.renderer.update_scene(self.mj_data)
            return self.renderer.render()
            
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            

class RLPathEnvVec(RLPathEnv):
    """
    向量化环境的基础版本,用于Stable-Baselines3的并行训练
    """
    pass


def make_rl_path_env(
    render_mode: Optional[str] = None,
    max_steps: int = 500,
    curriculum_level: int = 0,
    randomize_obstacles: bool = False,
) -> RLPathEnv:
    """
    创建RL路径规划环境的工厂函数
    
    Args:
        render_mode: 渲染模式 ("human", "rgb_array", None)
        max_steps: 每个episode的最大步数
        curriculum_level: 课程学习级别 (0=简单, 1=中等, 2=困难)
        randomize_obstacles: 是否随机化障碍物
        
    Returns:
        配置好的RLPathEnv实例
    """
    return RLPathEnv(
        render_mode=render_mode,
        max_steps=max_steps,
        curriculum_level=curriculum_level,
        randomize_obstacles=randomize_obstacles,
    )


if __name__ == "__main__":
    # 测试环境
    print("Testing RLPathEnv...")
    
    env = make_rl_path_env(render_mode=None, max_steps=100)
    
    # 测试reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs[:12]}...")
    print(f"Info: {info}")
    
    # 测试step
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, dist={info['dist_to_goal']:.4f}, done={terminated or truncated}")
        
        if terminated or truncated:
            break
            
    env.close()
    print("Test passed!")
