import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from common.motion_type import MotionType
from common.trajectory import Trajectory
from hallucination.hallucination import Hallucination

class HallucinationGenerator:
    def __init__(self):
        # 1. 直接定義參數 (不再依賴外部 Config)
        self.dt = 0.1
        self.history_time = 1.5
        self.future_time = 1.5

        # 自動計算步數
        self.history_steps = int(self.history_time / self.dt)
        self.future_steps = int(self.future_time / self.dt)

        # 2. 障礙物幾何參數
        self.wall_pillar_radius = 0.15  # 牆壁圓柱半徑
        self.wall_density = 0.2         # 牆壁間距
        self.robot_width = 0.5          # 機器人寬度 (用於計算安全通道)

    def process_trajectory(self, trajectory: Trajectory) -> List[Hallucination]:
        """
        處理整條軌跡，回傳多個 Frame 的幻覺資料。
        """
        generated_frames = []

        # --- Step 1: 生成一次性的靜態障礙物 (Static Map) ---
        all_poses = trajectory.states
        static_obstacles = self._generate_static_obstacles(trajectory, all_poses)

        # --- Step 2: 滑動視窗切分 Frame ---
        traj_len = len(trajectory)
        start_idx = self.history_steps
        end_idx = traj_len - self.future_steps

        for t in range(start_idx, end_idx):
            frame_data = trajectory.get_frame_data(t, self.history_steps, self.future_steps)
            if frame_data is None: continue

            (current_idx, current_pose, _, future_poses, last_cmd, target_cmd) = frame_data
            current_time = current_idx * self.dt

            # 計算 Local Goal
            goal_global = future_poses[-1, :2]
            local_goal = transform_to_local(current_pose, goal_global)

            # 封裝
            hallucination = Hallucination(
                frame_idx=current_idx,
                timestamp=current_time,
                pose_global=current_pose,
                current_vel=last_cmd,
                target_cmd=target_cmd,
                local_goal=local_goal,
                obstacles=static_obstacles
            )
            generated_frames.append(hallucination)

        plot_trajectory_and_obstacles(trajectory, generated_frames, title_suffix="(Static Obstacles)")
        return generated_frames

    # =========================================================================
    # 靜態地圖生成邏輯
    # =========================================================================

    def _generate_static_obstacles(self,
                                   traj: Trajectory,
                                   poses: np.ndarray) -> List[Tuple[float, float, float]]:

        m_type = traj.motion_type
        obstacles = []

        # --- A. Forward (直行 - 混合策略) ---
        if m_type == MotionType.forward:
            # 隨機決定環境類型：
            # 50% 機率是「走廊」(牆壁限制)
            # 50% 機率是「散落障礙物」(森林/雜物堆)
            if np.random.rand() > 0.5:
                # 模式 1: 寬走廊 (3.0m)
                obstacles.extend(self._create_corridor(poses, width=3.0))
            else:
                # 模式 2: 散落障礙物 (Scattered)
                # 這能訓練機器人在開放空間避障，而不只是沿著牆走
                obstacles.extend(self._create_scattered_obstacles(poses, density=0.5))

            return obstacles

        # --- B. Chicane (S型) ---
        if m_type == MotionType.chicane:
            obstacles.extend(self._create_corridor(poses, width=3.0))
            return obstacles

        # --- C. Threading (窄縫) ---
        if m_type == MotionType.threading:
            # 極窄走廊
            gap = self.robot_width + 0.4
            obstacles.extend(self._create_corridor(poses, width=gap))
            return obstacles

        # --- D. Turn (彎道) ---
        if m_type == MotionType.turn:
            # 判斷轉向 (計算整條路徑的角度變化)
            angle_diff = poses[-1, 2] - poses[0, 2]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            is_left_turn = angle_diff > 0

            # 內窄外寬
            inner_dist = 0.5
            outer_dist = 2.5
            dist_left = inner_dist if is_left_turn else outer_dist
            dist_right = outer_dist if is_left_turn else inner_dist

            obstacles.extend(self._create_asymmetric_corridor(poses, dist_left, dist_right))
            return obstacles

        # --- E. Brake (死路) ---
        if m_type == MotionType.brake:
            obstacles.extend(self._create_corridor(poses, width=3.0))
            # 終點封牆
            end_pose = poses[-1]
            obstacles.extend(self._create_perpendicular_wall(end_pose[:2], end_pose[2], width=4.0))
            return obstacles

        # --- F. Spin (盒狀空間) ---
        if m_type == MotionType.spin:
            center = poses[0][:2]
            radius = 1.5
            # 圓形圍牆
            num_obs = 12
            for i in range(num_obs):
                angle = (2 * np.pi / num_obs) * i
                ox = center[0] + radius * np.cos(angle)
                oy = center[1] + radius * np.sin(angle)
                obstacles.append((ox, oy, 0.3))
            return obstacles

        # --- G. Backward (貼臉牆) ---
        if m_type == MotionType.backward:
            start_x, start_y, start_th = poses[0]
            dist_in_front = 0.2
            wall_center = np.array([
                start_x + dist_in_front * np.cos(start_th),
                start_y + dist_in_front * np.sin(start_th)
            ])
            obstacles.extend(self._create_perpendicular_wall(wall_center, start_th, width=3.0))
            return obstacles

        return obstacles

    # =========================================================================
    # 幾何工具
    # =========================================================================

    def _create_corridor(self, poses, width):
        return self._create_asymmetric_corridor(poses, width/2.0, width/2.0)

    def _create_asymmetric_corridor(self, poses, left_dist, right_dist):
        walls = []
        step_size = 3
        for i in range(0, len(poses), step_size):
            x, y, th = poses[i]
            perp_x, perp_y = -np.sin(th), np.cos(th)

            # 左牆
            walls.append((x + perp_x * left_dist, y + perp_y * left_dist, self.wall_pillar_radius))
            # 右牆
            walls.append((x - perp_x * right_dist, y - perp_y * right_dist, self.wall_pillar_radius))
        return walls

    def _create_perpendicular_wall(self, center, heading, width):
        walls = []
        wall_vec_x, wall_vec_y = -np.sin(heading), np.cos(heading)
        num_pillars = int(width / (self.wall_pillar_radius * 1.5))
        for i in range(num_pillars):
            offset = (i / (num_pillars - 1) - 0.5) * width
            wx = center[0] + wall_vec_x * offset
            wy = center[1] + wall_vec_y * offset
            walls.append((wx, wy, self.wall_pillar_radius))
        return walls

    def _create_scattered_obstacles(self, poses, density: float) -> List[Tuple]:
        """
        沿著路徑周圍隨機生成障礙物，但確保路徑本身是「乾淨」的。

        :param density: 障礙物密度 (越高越密)
        """
        obstacles = []
        step_size = 5 # 不需要每個點都算，每隔幾步生成一組

        # 安全通道半徑 (保證機器人不會撞到生成的障礙物)
        safe_radius = self.robot_width * 1.2
        # 生成範圍 (只在路徑兩側 5米內生成)
        spawn_width = 5.0

        for i in range(0, len(poses), step_size):
            x, y, th = poses[i]
            perp_x, perp_y = -np.sin(th), np.cos(th)

            # 嘗試在兩側生成隨機數量的障礙物
            num_attempts = np.random.randint(1, 10) # 每次 1~3 個

            for _ in range(num_attempts):
                # 隨機距離：必須大於安全半徑，小於生成範圍
                dist = np.random.uniform(safe_radius, spawn_width)

                # 隨機左右
                side = 1 if np.random.rand() > 0.5 else -1

                # 計算位置
                obs_x = x + perp_x * dist * side
                obs_y = y + perp_y * dist * side

                # 加入一點隨機擾動 (前後位移)，讓它不要排得太整齊
                noise_x = np.random.uniform(-0.5, 0.5)
                noise_y = np.random.uniform(-0.5, 0.5)

                obstacles.append((obs_x + noise_x, obs_y + noise_y, 0.2)) # 半徑 0.2 的障礙物

        return obstacles

def normalize_angle(angle):
    """將角度標準化到 (-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def transform_to_local(robot_pose_global: np.ndarray, points_global: np.ndarray) -> np.ndarray:
    """
    將全局座標點轉換到機器人局部座標系 (Body Frame)。

    :param robot_pose_global: 機器人當前姿態 [x, y, theta]
    :param points_global: 全局點陣列，Shape (N, 2) 或 (2,)
    :return: 局部點陣列，Shape 與輸入相同
    """
    rx, ry, rtheta = robot_pose_global
    c = np.cos(rtheta)
    s = np.sin(rtheta)

    # 旋轉矩陣 (Global to Local) 是旋轉矩陣 R(theta) 的轉置(反矩陣)
    # R_inv = [ c  s]
    #         [-s  c]
    rotation_matrix_inv = np.array([[c, s], [-s, c]])

    # 1. 平移 (相對於機器人位置)
    points_translated = points_global - np.array([rx, ry])

    # 2. 旋轉
    # 如果輸入是單個點 (2,)
    if points_translated.ndim == 1:
        return rotation_matrix_inv @ points_translated
    # 如果輸入是多個點 (N, 2)
    else:
        # 利用矩陣乘法技巧: (R_inv @ P.T).T
        return (rotation_matrix_inv @ points_translated.T).T

def plot_trajectory_and_obstacles(trajectory: Trajectory,
                                      hallucination_frames: List[Hallucination],
                                      title_suffix: str = ""):
    """
    將完整的機器人軌跡與生成的靜態幻覺障礙物繪製在同一張圖上。

    :param trajectory: 完整的軌跡物件 (包含所有 poses 和 motion_type)
    :param hallucination_frames: HallucinationGenerator 生成的 Frame 列表。
                                    (我們只需要取第一個 frame 的 obstacles，因為它們是靜態共享的)
    :param title_suffix: 標題後綴 (選用)
    """
    if not hallucination_frames:
        print("Warning: No hallucination frames to plot.")
        return

    # 1. 準備數據
    poses = trajectory.states  # [N, 3] array
    # 由於採用靜態環境策略，所有 frame 共享同一組障礙物，取第一個即可
    static_obstacles = hallucination_frames[0].obstacles
    motion_type_name = trajectory.motion_type.name

    # 2. 建立圖表
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal') # 關鍵：確保幾何比例正確
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # --- 繪製障礙物 (Hallucinations) ---
    # 使用紅色半透明圓形
    print(f"Plotting {len(static_obstacles)} obstacles...")
    for i, (obs_x, obs_y, obs_r) in enumerate(static_obstacles):
        # zorder=2 確保障礙物在網格之上，軌跡之下
        circle = Circle((obs_x, obs_y), obs_r, color='red', alpha=0.5, ec='darkred', zorder=2)
        ax.add_artist(circle)
        # 只為第一個障礙物加標籤，避免圖例爆炸
        if i == 0:
            circle.set_label('Hallucinated Obstacles')

    # --- 繪製軌跡 (Trajectory) ---
    # 使用藍色實線
    ax.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=2.5, alpha=0.8, label='Robot Trajectory', zorder=3)

    # 標記起點 (綠色圓點)
    ax.plot(poses[0, 0], poses[0, 1], 'go', markersize=10, zorder=4, label='Start')
    # 標記終點 (紅色叉叉)
    ax.plot(poses[-1, 0], poses[-1, 1], 'rx', markersize=10, markeredgewidth=2, zorder=4, label='End')

    # 標記起始方向 (小箭頭)
    start_x, start_y, start_th = poses[0]
    ax.arrow(start_x, start_y,
                0.5 * np.cos(start_th), 0.5 * np.sin(start_th),
                head_width=0.2, head_length=0.3, fc='green', ec='green', zorder=5)


    # 4. 設定圖表屬性
    ax.set_title(f"Motion Type: [{motion_type_name.upper()}] {title_suffix}", fontsize=14)
    ax.set_xlabel("Global X (m)", fontsize=12)
    ax.set_ylabel("Global Y (m)", fontsize=12)
    ax.legend(loc='best', shadow=True)

    # 自動調整視野範圍並加入一點邊距 (Padding)
    ax.autoscale_view()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pad = 1.0
    ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
    ax.set_ylim(ylim[0] - pad, ylim[1] + pad)

    plt.tight_layout()
    print("Displaying plot...")
    plt.show()
