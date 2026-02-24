"""
Task-Space RL Environment for Robot Arm Path Planning
任务空间强化学习环境：从 pickup zone 到 drop zone 的末端路径规划

核心改进：
1. 观测空间：末端位置 + 姿态 + 关节角度 + 目标位置
2. 奖励函数：基于末端到目标距离
3. 成功判定：末端进入 drop zone
4. 自动复位：成功后返回 home 位置


"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import pinocchio


class RLTaskSpaceEnv(gym.Env):
    """
    任务空间强化学习环境
    
    任务：控制机械臂末端从 pickup zone 移动到 drop zone
    
    观测空间 (16维):
        - 末端执行器位置 (3,)
        - 末端执行器姿态四元数 (4,)
        - 当前关节位置 (6,)
        - 目标位置 (drop zone center) (3,)
    
    动作空间 (6维):
        - 关节位置增量 [-1, 1] → 缩放为实际增量
    
    奖励设计:
        - 末端到目标距离奖励 (dense)
        - 平滑性惩罚
        - 碰撞惩罚
        - 成功到达奖励
        - 时间惩罚
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
        visualize: bool = False,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.visualize = visualize
        
        # Zone 定义 (从 scene.xml 中获取)
        # zone_pickup: pos="1.4 0.6 0.73" size="0.2 0.6 0.01"
        # zone_drop: pos="0.6 0.2 0.73" size="0.2 0.2 0.01"
        self.pickup_zone_center = np.array([1.4, 0.6, 0.73])
        self.pickup_zone_size = np.array([0.2, 0.6, 0.01])
        self.drop_zone_center = np.array([0.6, 0.2, 0.83])
        self.drop_zone_size = np.array([0.2, 0.2, 0.01])
        
        # 起始点采样半径 (用户要求: 0.3m)
        self.start_sample_radius = 0.3
        
        # 成功阈值：末端到 drop zone 中心的距离
        self.success_threshold = 0.05  # 5cm
        
        # ============================================================
        # V3 改进奖励配置 (当前使用)
        # ============================================================
        self.reward_config = {
            "delta_weight": 100.0,        # 核心引导信号 (提高2倍)
            "proximity_bonus": 0.0,       # 接近奖励 (移除以避免局部最优)
            "smoothness_weight": 0.05,    # 平滑性惩罚
            "collision_penalty": -1000.0, # 碰撞惩罚 (严格禁止碰撞)
            "success_bonus": 1000.0,      # 成功奖励 (提高5倍)
            "time_penalty": -1.0,         # 时间惩罚 (提高惩罚)
            "joint_limit_penalty": 5.0,   # 关节限制惩罚 (用于 barrier function)
        }
        
        # ============================================================
        # 自适应奖励配置 (Curriculum Learning) - 已禁用
        # ============================================================
        # # 这些是"基础值"，会根据训练进度动态调整
        # self.reward_config = {
        #     # 引导类奖励 (早期强，后期弱)
        #     "delta_weight_base": 50.0,        # 距离差分奖励基础权重
        #     "proximity_bonus_base": 5.0,      # 接近奖励基础值
        #     
        #     # 目标类奖励 (早期弱，后期强)  
        #     "success_bonus_base": 100.0,      # 成功奖励基础值
        #     "distance_weight_base": 0.5,      # 距离惩罚基础权重
        #     
        #     # 固定惩罚 (不随训练变化)
        #     "smoothness_weight": 0.05,        # 平滑性惩罚
        #     "collision_penalty": -50.0,       # 碰撞惩罚
        #     "time_penalty": -0.2,             # 时间惩罚
        #     "joint_limit_penalty": 5.0,       # 关节限制惩罚 (用于 barrier function)
        # }
        # 
        # # 课程学习参数
        # # 完整课程需要 ~11M timesteps (55k episodes × 200 steps)
        # # 建议训练命令: --timesteps 15000000
        # self.curriculum_config = {
        #     "warmup_episodes": 5000,          # 预热阶段 (~1M timesteps)
        #     "curriculum_episodes": 50000,     # 课程过渡 (~10M timesteps)
        #     "delta_decay": 0.5,               # 训练结束时 delta_weight 衰减到 base * decay
        #     "success_growth": 2.0,            # 训练结束时 success_bonus 增长到 base * growth
        #     "distance_growth": 4.0,           # 训练结束时 distance_weight 增长到 base * growth
        # }
        
        # 机器人参数
        self.n_joints = 6
        self.action_scale = 0.3  # 每步最大移动 ~17度
        
        # 关节限制 (UR3e)
        self.joint_limits_low = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_high = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # 预设配置
        self.home_config = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
        
        # 加载模型
        self._load_mujoco_model()
        self._load_pinocchio_model()
        
        # 观测空间: ee_pos(3) + ee_quat(4) + joint_pos(6) + target_pos(3) = 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        
        # 动作空间: 6维关节增量
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
        
        # 状态变量
        self.current_step = 0
        self.target_ee_pos = None  # 采样的起始末端位置
        self.prev_action = np.zeros(self.n_joints)
        self.prev_ee_pos = None
        self.prev_ee_velocity = np.zeros(3)
        
        # 课程学习计数器 (已禁用)
        # self.total_episodes = 0
        
        # 可视化
        self.viewer = None
        self.start_time = None
        
        # 随机数生成器
        self.np_random = np.random.default_rng()
        
    def _load_mujoco_model(self):
        """加载MuJoCo模型"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scene_path = os.path.join(base_dir, 'assets', 'scenes', 'scene.xml')
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
            
        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
    def _load_pinocchio_model(self):
        """加载Pinocchio模型用于运动学计算"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(base_dir, 'robot_description', 'urdf', 'ur3e_ag95.urdf')
        
        if os.path.exists(urdf_path):
            self.pin_model = pinocchio.buildModelFromUrdf(urdf_path)
            self.pin_data = self.pin_model.createData()
            self.ee_frame_id = self.pin_model.getFrameId("grasp_center")
        else:
            self.pin_model = None
            self.pin_data = None
            self.ee_frame_id = None
            print(f"Warning: URDF not found at {urdf_path}, FK will be limited")
            
    def _get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """计算末端执行器位置"""
        if self.pin_model is not None:
            q_full = np.zeros(self.pin_model.nq)
            q_full[:6] = q
            pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
            pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
            return self.pin_data.oMf[self.ee_frame_id].translation.copy()
        else:
            # 使用 MuJoCo body position 作为备选
            try:
                ee_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
                return self.mj_data.xpos[ee_body_id].copy()
            except:
                return np.zeros(3)
            
    def _get_ee_orientation(self, q: np.ndarray) -> np.ndarray:
        """计算末端执行器姿态 (四元数)"""
        if self.pin_model is not None:
            q_full = np.zeros(self.pin_model.nq)
            q_full[:6] = q
            pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
            pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
            
            rotation = self.pin_data.oMf[self.ee_frame_id].rotation
            quat = pinocchio.Quaternion(rotation).coeffs()  # [x, y, z, w]
            return quat.copy()
        else:
            return np.array([0, 0, 0, 1])  # 默认姿态
            
    def _solve_ik(self, target_pos: np.ndarray, max_iter: int = 100, tol: float = 1e-4) -> np.ndarray:
        """
        使用 Jacobian 伪逆求解简单的 IK
        找到末端位置为 target_pos 的关节配置
        """
        if self.pin_model is None:
            return self.home_config.copy()
            
        # 从 home config 开始迭代
        q = self.home_config.copy()
        
        for _ in range(max_iter):
            # 正向运动学
            ee_pos = self._get_ee_position(q)
            error = target_pos - ee_pos
            
            if np.linalg.norm(error) < tol:
                break
                
            # 计算 Jacobian
            q_full = np.zeros(self.pin_model.nq)
            q_full[:6] = q
            pinocchio.computeJointJacobians(self.pin_model, self.pin_data, q_full)
            J = pinocchio.getFrameJacobian(
                self.pin_model, self.pin_data, 
                self.ee_frame_id, 
                pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3, :6]  # 只取线速度部分 (3,) 和前6个关节 (:6)
            
            # 使用阻尼最小二乘法 (Damped Least Squares)
            lambda_val = 0.05
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_val**2 * np.eye(3))
            
            # 更新关节角
            dq = J_pinv @ error
            q = q + dq * 0.5  # 步长 0.5
            
            # 关节限制
            q = np.clip(q, self.joint_limits_low, self.joint_limits_high)
            
        return q
            
    def _sample_start_position(self) -> np.ndarray:
        """
        在 pickup zone 中心附近采样起始末端位置
        半径 0.3m 的圆圈内随机采样
        """
        # 在 xy 平面上采样
        angle = self.np_random.uniform(0, 2 * np.pi)
        radius = self.np_random.uniform(0, self.start_sample_radius)
        
        x = self.pickup_zone_center[0] + radius * np.cos(angle)
        y = self.pickup_zone_center[1] + radius * np.sin(angle)
        z = self.pickup_zone_center[2] + 0.05  # 稍微抬高，避免与桌面碰撞
        
        return np.array([x, y, z])
        
    def _check_collision(self) -> bool:
        """检查机器人是否发生碰撞"""
        if self.mj_data.ncon == 0:
            return False
            
        # 可视化标记（永远忽略）
        visualization_geoms = {
            'floor',  # 地面可以接触
            'x-aixs', 'y-aixs', 'z-aixs',  # 坐标轴标记
            'mocap',  # mocap标记
            'zone_pickup', 'zone_drop',  # 区域可视化标记
        }
        
        # 场景物体（物体间碰撞忽略，但与机器人碰撞要检测）
        scene_objects = {
            'Apple', 'Banana',  # 场景中的物体
            'simple_table', 'table1', 'table2',  # 桌子
        }
        
        visualization_ids = set()
        scene_object_ids = set()
        
        for i in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                if name in visualization_geoms:
                    visualization_ids.add(i)
                elif name in scene_objects:
                    scene_object_ids.add(i)
                
        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            
            # 忽略可视化标记
            if geom1 in visualization_ids or geom2 in visualization_ids:
                continue
            
            # 忽略场景物体之间的碰撞（如Banana和table）
            if geom1 in scene_object_ids and geom2 in scene_object_ids:
                continue
                
            # 忽略微小接触（与场景物体）
            if geom1 in scene_object_ids or geom2 in scene_object_ids:
                if contact.dist > -0.005:  # 穿透小于5mm忽略
                    continue
                    
            # 其他情况视为碰撞（机器人与障碍物）
            if contact.dist < -0.01:
                return True
                
        return False
        
    def _check_joint_limits(self, q: np.ndarray) -> bool:
        """检查是否超出关节限制"""
        return np.any(q < self.joint_limits_low) or np.any(q > self.joint_limits_high)
        
    def _compute_joint_limit_penalty(self, q: np.ndarray) -> float:
        """
        计算渐进式关节限位惩罚 (Barrier Function)
        
        当关节角度接近限位时，惩罚逐渐增大：
        - 在安全区域 (0-80% 范围) 内：无惩罚
        - 超过 80% 范围：惩罚随距离平方增长
        - 超出限位：最大惩罚
        
        Returns:
            float: 负数惩罚值 (越接近限位，值越负)
        """
        penalty = 0.0
        margin_ratio = 0.2  # 20% 的范围作为警戒区
        
        for i in range(self.n_joints):
            q_min = self.joint_limits_low[i]
            q_max = self.joint_limits_high[i]
            q_range = q_max - q_min
            margin = q_range * margin_ratio
            
            # 计算到安全边界的距离
            safe_min = q_min + margin
            safe_max = q_max - margin
            
            if q[i] < safe_min:
                # 接近下限
                dist_to_margin = (safe_min - q[i]) / margin
                penalty += dist_to_margin ** 2
            elif q[i] > safe_max:
                # 接近上限
                dist_to_margin = (q[i] - safe_max) / margin
                penalty += dist_to_margin ** 2
                
        return -penalty * self.reward_config["joint_limit_penalty"]
        
    def _check_in_drop_zone(self, ee_pos: np.ndarray) -> bool:
        """检查末端是否到达目标 (使用欧氏距离)"""
        # 使用球形距离阈值，不是box！
        dist_to_target = np.linalg.norm(ee_pos - self.drop_zone_center)
        return dist_to_target < self.success_threshold  # 0.05m
    
    # ============================================================
    # 课程学习进度计算 (已禁用)
    # ============================================================
    # def _get_curriculum_progress(self) -> float:
    #     """
    #     计算课程学习进度 (0.0 ~ 1.0)
    #     
    #     Returns:
    #         progress: 0.0 = 训练开始, 1.0 = 课程结束
    #     """
    #     warmup = self.curriculum_config["warmup_episodes"]
    #     curriculum = self.curriculum_config["curriculum_episodes"]
    #     
    #     if self.total_episodes < warmup:
    #         return 0.0  # 预热阶段，使用初始权重
    #     
    #     episodes_in_curriculum = self.total_episodes - warmup
    #     progress = min(1.0, episodes_in_curriculum / curriculum)
    #     return progress
        
    def _compute_reward(
        self,
        ee_pos: np.ndarray,
        action: np.ndarray,
        collision: bool,
        reached_goal: bool
    ) -> Tuple[float, Dict]:
        """
        计算奖励 - V3 版本奖励函数
        
        核心思路:
        1. Delta reward: 引导 agent 靠近目标
        2. Proximity bonus: 奖励接近目标的状态
        3. Success bonus: 奖励到达目标
        4. 无距离惩罚: 避免全负奖励影响训练初期
        """
        info = {}
        reward = 0.0
        cfg = self.reward_config
        
        # 1. 计算当前距离
        dist_to_target = np.linalg.norm(ee_pos - self.drop_zone_center)
        info["dist_to_target"] = dist_to_target
        
        # 2. 距离差分奖励 (核心引导信号)
        if self.prev_ee_pos is not None:
            prev_dist = np.linalg.norm(self.prev_ee_pos - self.drop_zone_center)
            delta_dist = prev_dist - dist_to_target  # 正值 = 更近
            delta_reward = delta_dist * cfg["delta_weight"]
            reward += delta_reward
            info["delta_reward"] = delta_reward
        else:
            info["delta_reward"] = 0.0
            
        # 3. 接近目标额外奖励 (已移除以避免局部最优)
        # 如果需要启用，仅在极近距离(<0.05m)给予
        if cfg["proximity_bonus"] > 0 and dist_to_target < 0.05:
            proximity_bonus = cfg["proximity_bonus"]
        else:
            proximity_bonus = 0.0
        reward += proximity_bonus
        info["proximity_bonus"] = proximity_bonus
            
        # 4. 成功奖励
        if reached_goal:
            reward += cfg["success_bonus"]
            info["success_bonus"] = cfg["success_bonus"]
        else:
            info["success_bonus"] = 0.0
            
        # 5. 平滑性惩罚
        action_magnitude = np.linalg.norm(action)
        smoothness_penalty = -action_magnitude * cfg["smoothness_weight"]
        reward += smoothness_penalty
        info["smoothness_penalty"] = smoothness_penalty
            
        # 6. 碰撞惩罚
        if collision:
            reward += cfg["collision_penalty"]
            info["collision"] = True
        else:
            info["collision"] = False
            
        # 7. 时间惩罚
        reward += cfg["time_penalty"]
        
        # 8. 渐进式关节限位惩罚 (Barrier Function)
        q = self.mj_data.qpos[:6].copy()
        joint_limit_penalty = self._compute_joint_limit_penalty(q)
        reward += joint_limit_penalty
        info["joint_limit_penalty"] = joint_limit_penalty
        info["joint_limit_violation"] = self._check_joint_limits(q)
            
        info["success"] = reached_goal
        
        return reward, info
        
    def _get_observation(self) -> np.ndarray:
        """构建观测向量"""
        q_current = self.mj_data.qpos[:6].copy()
        
        # 末端位置和姿态
        ee_pos = self._get_ee_position(q_current)
        ee_quat = self._get_ee_orientation(q_current)
        
        # 目标位置 (drop zone center)
        target_pos = self.drop_zone_center.copy()
        
        obs = np.concatenate([
            ee_pos,      # 3
            ee_quat,     # 4
            q_current,   # 6
            target_pos,  # 3
        ])
        
        return obs.astype(np.float32)
        
    def _render_markers(self):
        """在 MuJoCo 可视化中渲染标记点"""
        if self.viewer is None:
            return
            
        # 清除之前的标记
        self.viewer.user_scn.ngeom = 0
        
        # 渲染目标点 (drop zone center) - 红色球
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=self.drop_zone_center,
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 0.8], dtype=np.float32)
        )
        self.viewer.user_scn.ngeom = 1
        
        # 渲染起始采样点 (如果有) - 绿色球
        if self.target_ee_pos is not None:
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[1],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=self.target_ee_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 1, 0, 0.8], dtype=np.float32)
            )
            self.viewer.user_scn.ngeom = 2
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            
        options = options or {}
        
        # 1. 设置目标位置
        if "target_ee_pos" in options:
            self.target_ee_pos = np.array(options["target_ee_pos"])
        else:
            # 默认采样
            self.target_ee_pos = self._sample_start_position()
            
        # 2. 设置初始关节角度
        if "initial_qpos" in options:
            q_start = np.array(options["initial_qpos"])
        else:
            # 使用 IK 计算对应的关节配置
            q_start = self._solve_ik(self.target_ee_pos)
        
        # 重置 MuJoCo 状态
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:6] = q_start
        self.mj_data.ctrl[:6] = q_start
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 初始化状态变量
        self.current_step = 0
        self.prev_action = np.zeros(self.n_joints)
        self.prev_ee_pos = self._get_ee_position(q_start)
        self.prev_ee_velocity = np.zeros(3)
        self.start_time = time.time()
        
        # 更新课程学习计数器 (已禁用)
        # self.total_episodes += 1
        
        # 初始化/更新可视化
        if self.visualize and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self.viewer.cam.distance = 2.0
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -30
            self.viewer.cam.lookat = np.array([1.0, 0.5, 0.8])
            
        if self.visualize and self.viewer is not None:
            self._render_markers()
            self.viewer.sync()
            
        obs = self._get_observation()
        info = {
            "target_ee_pos": self.target_ee_pos.copy(),
            "drop_zone": self.drop_zone_center.copy(),
        }
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步动作"""
        self.current_step += 1
        
        # 动作缩放
        scaled_action = action * self.action_scale
        
        # 当前关节位置
        q_current = self.mj_data.qpos[:6].copy()
        
        # 目标关节位置
        q_target = q_current + scaled_action
        q_target = np.clip(q_target, self.joint_limits_low, self.joint_limits_high)
        
        # 设置控制目标并仿真
        self.mj_data.ctrl[:6] = q_target
        
        n_substeps = 10
        for _ in range(n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)
            
        # 获取实际状态
        q_actual = self.mj_data.qpos[:6].copy()
        ee_pos = self._get_ee_position(q_actual)
        
        # 检查终止条件
        collision = self._check_collision()
        reached_goal = self._check_in_drop_zone(ee_pos)
        timeout = self.current_step >= self.max_steps
        
        # 计算奖励
        reward, info = self._compute_reward(ee_pos, action, collision, reached_goal)
        
        # 终止条件
        terminated = reached_goal or collision
        truncated = timeout
        
        # 更新状态
        self.prev_action = action.copy()
        self.prev_ee_pos = ee_pos.copy()
        
        # 添加信息
        info["step"] = self.current_step
        info["ee_pos"] = ee_pos.copy()
        info["reached_goal"] = reached_goal
        info["timeout"] = timeout
        
        # 可视化
        if self.visualize and self.viewer is not None:
            self._render_markers()
            self.viewer.sync()
            time.sleep(0.01)  # 减慢可视化速度
            
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
        
    def set_curriculum_level(self, level: int):
        """设置课程学习级别 (兼容接口)"""
        pass  # 暂不实现课程学习
        
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self._render_markers()
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
            renderer.update_scene(self.mj_data)
            return renderer.render()
            
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            

def make_task_space_env(
    render_mode: Optional[str] = None,
    max_steps: int = 200,
    visualize: bool = False,
) -> RLTaskSpaceEnv:
    """创建任务空间环境的工厂函数"""
    return RLTaskSpaceEnv(
        render_mode=render_mode,
        max_steps=max_steps,
        visualize=visualize,
    )


if __name__ == "__main__":
    # 测试环境
    print("Testing RLTaskSpaceEnv...")
    
    env = make_task_space_env(visualize=True, max_steps=100)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial EE position: {obs[:3]}")
    print(f"Target position: {obs[13:16]}")
    print(f"Info: {info}")
    
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample() * 0.3
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i+1}: reward={reward:.2f}, dist={info['dist_to_target']:.4f}")
            
        if terminated or truncated:
            print(f"Episode ended: success={info.get('success', False)}")
            break
            
    print(f"Total reward: {total_reward:.2f}")
    
    input("Press Enter to close...")
    env.close()
    print("Test completed!")
