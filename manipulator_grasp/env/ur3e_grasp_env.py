import os.path
import sys

sys.path.append('../../manipulator_grasp')

import numpy as np
import mujoco
import mujoco.viewer

import glfw
import cv2
from typing import Tuple
from scipy.spatial.transform import Rotation


from manipulator_grasp.path_plan.set_model import (
    load_models,
    add_self_collisions,
    add_object_collisions,
    load_path_planner,
)


class UR3eGraspEnv:

    def __init__(self):
        self.sim_hz = 500

        self.model: mujoco.MjModel = None
        self.data: mujoco.MjData = None

        self.model_roboplan = None
        self.collision_model = None
        self.data_roboplan = None
        self.target_frame = None
        self.ik = None
        self.rrt_options = None

        self.renderer: mujoco.Renderer = None
        self.depth_renderer: mujoco.Renderer = None
        self.viewer: mujoco.viewer.Handle = None

        self.height = 640 # 256 640 720
        self.width = 640 # 256 640 1280
        self.fovy = np.pi / 4

        # 新增离屏渲染相关属性
        self.camera_name = "cam"
        self.camera_id = -1
        self.offscreen_context = None
        self.offscreen_scene = None
        self.offscreen_camera = None
        self.offscreen_viewport = None
        self.glfw_window = None

    def reset(self):
        # 初始化路径规划模型
        urdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robot_description', 'urdf', 'ur3e_ag95.urdf')
        srdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robot_description', 'srdf', 'ur3e_ag95.srdf')
        self.model_roboplan, self.collision_model, visual_model = load_models(urdf_path)
        add_self_collisions(self.model_roboplan, self.collision_model, srdf_path)
        add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.04)

        self.data_roboplan = self.model_roboplan.createData()

        self.target_frame, self.ik, self.rrt_options = load_path_planner(self.model_roboplan, self.data_roboplan, self.collision_model)
        
        # 初始化 MuJoCo 模型和数据
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(filename)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        self.data.ctrl[:6] = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

        # 创建两个渲染器实例，分别用于生成彩色图像和深度图
        self.renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        self.depth_renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        # 更新渲染器中的场景数据
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        # 启用深度渲染
        self.depth_renderer.enable_depth_rendering()
        
        # 初始化被动查看器
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # 为了方便观察
        self.viewer.cam.lookat[:] = [1.8, 1.1, 1.7]  # 对应XML中的center
        self.viewer.cam.azimuth = 210      # 对应XML中的azimuth
        self.viewer.cam.elevation = -35    # 对应XML中的elevation
        self.viewer.cam.distance = 1.2     # 根据场景调整的距离值
        self.viewer.sync() # 立即同步更新

        # # --- 新增: 初始化离屏渲染 ---
        # # 初始化GLFW用于离屏渲染
        # glfw.init()
        # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        # self.glfw_window = glfw.create_window(self.width, self.height, "Offscreen", None, None)
        # glfw.make_context_current(self.glfw_window)

        # # 获取相机ID
        # self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        # if self.camera_id != -1:
        #     print(f"成功找到相机 '{self.camera_name}', ID: {self.camera_id}")
        #     # 使用XML中定义的固定相机
        #     self.offscreen_camera = mujoco.MjvCamera()
        #     mujoco.mjv_defaultCamera(self.offscreen_camera)
        #     self.offscreen_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        #     self.offscreen_camera.fixedcamid = self.camera_id

        # # 创建离屏场景和上下文
        # self.offscreen_scene = mujoco.MjvScene(self.model, maxgeom=10000)
        # self.offscreen_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # self.offscreen_viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        # mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.offscreen_context)

        # # 创建OpenCV窗口
        # cv2.namedWindow('MuJoCo Camera Output', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('MuJoCo Camera Output', self.width, self.height)


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.depth_renderer is not None:
            self.depth_renderer.close()

        # 清理离屏渲染资源
        cv2.destroyAllWindows()
        if self.glfw_window is not None:
            glfw.destroy_window(self.glfw_window)
        glfw.terminate()
        self.offscreen_context = None
        self.offscreen_scene = None

    def step(self, action=None):
        if action is not None:
            self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        self.viewer.sync()

        # # --- 新增: 离屏渲染和显示 ---
        # if all([self.offscreen_context, self.offscreen_scene, self.offscreen_camera]):
        #     # 更新场景
        #     mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), 
        #                          mujoco.MjvPerturb(), self.offscreen_camera, 
        #                          mujoco.mjtCatBit.mjCAT_ALL.value, self.offscreen_scene)
            
        #     # 渲染到离屏缓冲区
        #     mujoco.mjr_render(self.offscreen_viewport, self.offscreen_scene, self.offscreen_context)
            
        #     # 读取像素数据
        #     rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        #     mujoco.mjr_readPixels(rgb, None, self.offscreen_viewport, self.offscreen_context)
            
        #     # 转换颜色空间并显示
        #     bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        #     cv2.imshow('MuJoCo Camera Output', bgr)
            
        #     # 检查ESC键
        #     if cv2.waitKey(1) == 27:
        #         print("用户按下了ESC键,退出仿真。")
        #         self.close()
        #         exit(0)
                
    def render(self):
        '''
        常用于强化学习或机器人控制任务中，提供环境的视觉观测数据。
        '''
        # 更新渲染器中的场景数据
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        # 渲染图像和深度图
        return {
            'img': self.renderer.render(),
            'depth': self.depth_renderer.render()
        }

    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定 site 的位姿信息。
        
        Args:
            site_name (str): site 名称字符串
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - position (np.ndarray): 形状为 (3,) 的位置向量 [x, y, z]
                - quaternion (np.ndarray): 形状为 (4,) 的四元数向量 [w, x, y, z]
                
        Raises:
            ValueError: 如果找不到指定名称的 site
        """
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 '{site_name}' 的 site")

        position = np.array(self.data.site(site_id).xpos)
        xmat = np.array(self.data.site(site_id).xmat)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)

        return position, quaternion

    def get_body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定 body 的位姿信息。
        
        Args:
            body_name (str): body 名称字符串
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - position (np.ndarray): 形状为 (3,) 的位置向量 [x, y, z]
                - quaternion (np.ndarray): 形状为 (4,) 的四元数向量 [w, x, y, z]
                
        Raises:
            ValueError: 如果找不到指定名称的 body
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的 body")
        
        position = np.array(self.data.body(body_id).xpos)
        quaternion = np.array(self.data.body(body_id).xquat)
        
        return position, quaternion

    def check_collision(self, geom1_id: int, geom2_id: int) -> bool:
        """
        基于 MuJoCo 的原生接触数据检测两个几何体是否发生物理碰撞。
        比使用 AABB 包围盒或者距离计算更精确，这直接利用了底层解算器的碰撞流形。
        
        Args:
            geom1_id (int): 参与碰撞检测的第一个几何体 ID
            geom2_id (int): 参与碰撞检测的第二个几何体 ID
            
        Returns:
            bool: 如果发生碰撞返回 True，否则返回 False
        """
        for contact in self.data.contact:
            if (contact.geom1 == geom1_id and contact.geom2 == geom2_id) or \
               (contact.geom2 == geom1_id and contact.geom1 == geom2_id):
                return True
        return False

    def compute_reward(self, gripper_pos: np.ndarray, gripper_quat: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray) -> float:
        """
        计算平滑的抓取奖励（基于位置和姿态的误差）。
        可以通过 tanh 函数将无界的距离/角度差异拉伸到 [0, MAX] 区域内的连续得分。
        可用于评估 RL 动作或对各种算法生成的抓取位姿质量进行二次打分。
        
        Args:
            gripper_pos (np.ndarray): 夹爪末端位置 [x, y, z]
            gripper_quat (np.ndarray): 夹爪末端四元数 [w, x, y, z] (SciPy标准)
            target_pos (np.ndarray): 目标位置 [x, y, z]
            target_quat (np.ndarray): 目标四元数 [w, x, y, z] (SciPy标准)
            
        Returns:
            float: 连续且平滑的奖励值
        """
        # 1. 相对位置惩罚/奖励转换
        rel_pos = target_pos - gripper_pos
        distance = np.linalg.norm(rel_pos)
        # 距离越小，奖励越接近 5
        pos_reward = 5.0 * (1.0 - np.tanh(10.0 * distance))
        
        # 2. 相对姿态惩罚/奖励转换
        # 计算将当前姿态旋转到目标姿态所需的旋转向量
        rel_rot_vec = (Rotation.from_quat(gripper_quat).inv() * 
                       Rotation.from_quat(target_quat)).as_rotvec()
        rot_error = np.linalg.norm(rel_rot_vec)
        # 姿态误差越小，奖励越接近 1
        ori_reward = 1.0 * (1.0 - np.tanh(3.0 * rot_error))
        
        # 返回加和的总奖励
        return pos_reward + ori_reward



if __name__ == '__main__':
    env = UR3eGraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
