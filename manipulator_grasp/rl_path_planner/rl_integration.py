"""
RL Path Planner Integration Module
Provides an interface to use trained PPO model for path planning in grasp_process.py
"""

import os
import logging
import numpy as np
import pinocchio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import mujoco

logger = logging.getLogger(__name__)


class RLPathPlanner:
    """
    Wrapper class for using a trained PPO model for path planning.
    
    Supports both place phase (default) and approach phase models.
    Model paths are managed via model_config.py for easy switching.
    """
    
    def __init__(self, model_path: str = None, vecnormalize_path: str = None, phase: str = 'place',
                 action_scale: float = None, substeps: int = None):
        """
        Initialize the RL planner.
        
        Args:
            model_path: Path to the .zip model file. If None, uses config default.
            vecnormalize_path: Path to vecnormalize .pkl file. If None, auto-inferred.
            phase: 'place' or 'approach' - which phase model to use
            action_scale: Scale factor for actions (lower = slower). Default: 0.15
            substeps: MuJoCo substeps per RL step (higher = smoother). Default: 15
        """
        # Import config
        from . import model_config
        
        # Use config if no path provided
        if model_path is None:
            if phase == 'place':
                config = model_config.get_place_phase_config()
                model_path = config['model_path']
                if vecnormalize_path is None:
                    vecnormalize_path = config['vecnormalize_path']
                self.training_target = np.array(config['drop_zone_center'])
                self.success_threshold = config['success_threshold']
                self.max_steps = config['max_steps']
            elif phase == 'approach':
                config = model_config.get_approach_phase_config()
                model_path = config['model_path']
                if vecnormalize_path is None:
                    vecnormalize_path = config['vecnormalize_path']
                self.training_target = np.array(config['drop_zone_center'])
                self.success_threshold = config['success_threshold']
                self.max_steps = config['max_steps']
            else:
                raise ValueError(f"Unknown phase: {phase}. Use 'place' or 'approach'")
        else:
            # Manual path provided - use default parameters
            self.training_target = np.array([0.6, 0.2, 0.83])
            self.success_threshold = 0.10
            self.max_steps = 500
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Infer vecnormalize path if not provided (only if not already set by config)
        if vecnormalize_path is None:
            vecnormalize_path = os.path.join(
                os.path.dirname(model_path), 
                'final_model_vecnormalize.pkl'
            )
        
        print(f"[RL Planner] Loading model from {model_path}")
        self.model = PPO.load(model_path)
        
        # Load normalization statistics
        self.obs_mean = None
        self.obs_var = None
        if os.path.exists(vecnormalize_path):
            print(f"[RL Planner] Loading normalization from {vecnormalize_path}")
            import pickle
            with open(vecnormalize_path, 'rb') as f:
                vecnorm_data = pickle.load(f)
                if hasattr(vecnorm_data, 'obs_rms'):
                    self.obs_mean = vecnorm_data.obs_rms.mean
                    self.obs_var = vecnorm_data.obs_rms.var
                    print(f"[RL Planner] Loaded obs normalization: mean shape {self.obs_mean.shape}")
        else:
            print(f"[RL Planner] Warning: VecNormalize not found at {vecnormalize_path}")
        

        # RL environment parameters (with speed control)
        ### action_scale 决定了每一步 RL 动作给予底层 PD 控制器的目标关节增量（越小每次移动的角度越少）。
        ### substeps 决定了这一个增量经过多少次底层的物理演算来到达（越少插值时间越短，动作越快）。
        ### 要留意这两个数值是经典的 RL 机械臂控制参数组合。
        self.action_scale = action_scale if action_scale is not None else 0.15

        self.substeps = substeps if substeps is not None else 10
        self.clip_obs = 10.0  # VecNormalize uses clip_obs=10.0
        
        # Joint limits matching training environment (UR3e)
        self.joint_limits_low = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_high = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # Robot link name prefixes for collision detection
        self._robot_link_prefixes = ('ag95', 'link', 'left', 'right', 'shoulder', 'elbow', 'wrist', 'forearm', 'upper_arm')
        # Geom pairs to ignore in collision detection (e.g. base-ground)
        self._ignored_collision_pairs = {
            frozenset({'base_link_inertia', 'ground_plane'}),
        }
        
        logger.info(f"Speed config: action_scale={self.action_scale}, substeps={self.substeps}")
        
        # Pinocchio model for FK (will be set from env)
        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None
        
    def _setup_pinocchio(self, env):
        """Setup Pinocchio model from the grasp environment."""
        if self.pin_model is None and hasattr(env, 'model_roboplan'):
            self.pin_model = env.model_roboplan
            self.pin_data = self.pin_model.createData()
            self.ee_frame_id = self.pin_model.getFrameId("grasp_center")
            print(f"[RL Planner] Setup Pinocchio, EE frame ID: {self.ee_frame_id}")
    
    def _get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector position using Pinocchio FK."""
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not initialized. Call _setup_pinocchio first.")
        
        q_full = np.zeros(self.pin_model.nq)
        q_full[:6] = q
        pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
        return self.pin_data.oMf[self.ee_frame_id].translation.copy()
    
    def _get_ee_orientation(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector orientation as quaternion."""
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not initialized.")
        
        q_full = np.zeros(self.pin_model.nq)
        q_full[:6] = q
        pinocchio.forwardKinematics(self.pin_model, self.pin_data, q_full)
        pinocchio.updateFramePlacements(self.pin_model, self.pin_data)
        
        rotation = self.pin_data.oMf[self.ee_frame_id].rotation
        quat = pinocchio.Quaternion(rotation).coeffs()  # [x, y, z, w]
        return quat.copy()
    
    def _make_observation(self, env, target_pos: np.ndarray) -> np.ndarray:
        """
        Construct observation vector matching the RL training environment.
        
        Observation (16-dim):
            - ee_pos (3)
            - ee_quat (4)
            - joint_pos (6)
            - target_pos (3)
        """
        q_current = env.data.qpos[:6].copy()
        ee_pos = self._get_ee_position(q_current)
        ee_quat = self._get_ee_orientation(q_current)
        
        obs = np.concatenate([
            ee_pos,       # 3
            ee_quat,      # 4
            q_current,    # 6
            target_pos,   # 3
        ]).astype(np.float32)
        
        return obs
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply observation normalization if available (matching VecNormalize)."""
        if self.obs_mean is not None and self.obs_var is not None:
            normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            # VecNormalize clips to [-clip_obs, clip_obs]
            return np.clip(normalized, -self.clip_obs, self.clip_obs)
        return obs
    
    def _check_collision(self, env) -> bool:
        """
        检查机器人是否与环境发生碰撞。
        
        与训练环境 (RLPlaceEnv._check_collision) 保持一致，
        只检测涉及机器人 link 的碰撞，忽略场景物体之间的接触。
        
        Args:
            env: UR3eGraspEnv instance
            
        Returns:
            bool: True 表示检测到碰撞
        """
        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            geom1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1_name is None or geom2_name is None:
                continue
            
            # 检查是否涉及机器人 link
            geom1_is_robot = any(prefix in geom1_name for prefix in self._robot_link_prefixes)
            geom2_is_robot = any(prefix in geom2_name for prefix in self._robot_link_prefixes)
            
            if not (geom1_is_robot or geom2_is_robot):
                continue  # 跳过不涉及机器人的接触
            
            # 检查是否被忽略的碰撞对
            pair = frozenset({geom1_name, geom2_name})
            if pair in self._ignored_collision_pairs:
                continue
            
            # 检查碰撞力度
            if contact.dist < 0:  # 负距离表示穿透
                logger.warning(f"Collision detected: {geom1_name} <-> {geom2_name} (dist={contact.dist:.4f})")
                return True
        
        return False

    def run_plan(
        self, 
        env, 
        target_pos: np.ndarray,
        visualize: bool = True
    ) -> tuple:
        """
        Execute RL policy to move the robot to target position.
        
        Args:
            env: UR3eGraspEnv instance
            target_pos: Target end-effector position (3,)
            visualize: Whether to sync viewer during execution
            
        Returns:
            success: Whether the target was reached
            trajectory: List of joint configurations visited
        """
        self._setup_pinocchio(env)
        
        # IMPORTANT: Use the exact training target, not the requested target
        # The model was trained with a fixed target and doesn't generalize
        actual_target = self.training_target.copy()
        requested_target = np.array(target_pos)
        
        if np.linalg.norm(actual_target - requested_target) > 0.15:
            logger.warning(
                f"Requested target {requested_target} differs from "
                f"training target {actual_target}"
            )
        
        trajectory = []
        
        logger.info(f"Starting plan to training target: {actual_target}")
        logger.info(f"(Requested target was: {requested_target})")
        
        for step in range(self.max_steps):
            # Get current state
            q_current = env.data.qpos[:6].copy()
            trajectory.append(q_current.copy())
            
            # Check if reached training target
            ee_pos = self._get_ee_position(q_current)
            dist = np.linalg.norm(ee_pos - actual_target)
            
            if step % 20 == 0:
                logger.info(f"Step {step}: dist={dist:.4f}, ee={ee_pos}")
            
            if dist < self.success_threshold:
                logger.info(f"Target reached in {step} steps!")
                return True, trajectory
            
            # Collision detection — abort if collision (matching training behavior)
            if self._check_collision(env):
                logger.warning(f"Collision at step {step}, aborting RL plan.")
                return False, trajectory
            
            # Construct observation with TRAINING target (not requested)
            obs = self._make_observation(env, actual_target)
            obs_normalized = self._normalize_obs(obs)
            
            # Get action from policy
            action, _ = self.model.predict(obs_normalized, deterministic=True)
            
            # Scale action to joint deltas (matches training action_scale)
            scaled_action = action * self.action_scale
            
            # Apply action with correct joint limits (matching training)
            q_target = q_current + scaled_action
            q_target = np.clip(q_target, self.joint_limits_low, self.joint_limits_high)
            
            # Set control and step simulation
            env.data.ctrl[:6] = q_target
            
            # Substeps matching training environment
            for _ in range(self.substeps):
                mujoco.mj_step(env.model, env.data)
                if visualize and hasattr(env, 'viewer') and env.viewer is not None:
                    env.viewer.sync()
        
        logger.warning(f"Timeout after {self.max_steps} steps (dist={dist:.4f})")
        return False, trajectory


# Singleton instances for caching (one per phase)
_rl_planner_instances = {}

def get_rl_planner(model_path: str = None, phase: str = 'place') -> RLPathPlanner:
    """
    Get or create a cached RLPathPlanner instance.
    
    Args:
        model_path: Optional custom model path
        phase: 'place' or 'approach' - which phase model to use
    
    Returns:
        Cached RLPathPlanner instance for the specified phase
    """
    global _rl_planner_instances
    
    # Use phase as cache key if using default config, otherwise use model_path
    cache_key = model_path if model_path is not None else f"default_{phase}"
    
    if cache_key not in _rl_planner_instances:
        print(f"[RL Planner] Creating new planner instance for phase={phase}")
        _rl_planner_instances[cache_key] = RLPathPlanner(model_path, phase=phase)
    
    return _rl_planner_instances[cache_key]
