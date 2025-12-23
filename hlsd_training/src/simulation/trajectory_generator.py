import numpy as np
import logging
from common.motion_type import MotionType
from common.trajectory import Trajectory
from simulation.jackal_simulator import JackalSimulator

class TrajectoryGenerator:
    """
    Generating trajectories for the Jackal robot using the simulator.
    """
    def __init__(self, simulator: JackalSimulator):
        self.logger = logging.getLogger(__name__)
        self.sim = simulator

    def generate_single(self, motion_type: MotionType = None) -> Trajectory:
        # --- A. Forward (直行) ---
        if motion_type == MotionType.forward:
            # 距離：短衝刺 (2m) ~ 長巡航 (8m)
            dist = np.random.uniform(2.0, 8.0)
            # 速度：低速蠕行 (0.5) ~ 極速 (MAX_V)
            speed = np.random.uniform(0.5, self.sim.cfg.max_linear_vel)
            return self.generate_forward(dist, speed)

        # --- B. Turn (轉彎 - 最複雜的參數) ---
        elif motion_type == MotionType.turn:
            # 方向：隨機左或右
            direction = 1 if np.random.rand() > 0.5 else -1

            # 角度：45度 (切彎) ~ 135度 (大迴轉)
            angle = np.deg2rad(np.random.uniform(45, 135)) * direction

            # 半徑：0.6m (急彎) ~ 3.0m (高速彎)
            radius = np.random.uniform(0.6, 3.0)

            # 速度限制：計算物理極限 v <= w_max * R
            max_phys_speed = self.sim.cfg.max_angular_vel * radius
            limit_speed = min(self.sim.cfg.max_linear_vel, max_phys_speed)
            # 在極限內隨機取速，偏好較高的速度以增加難度
            target_speed = np.random.uniform(limit_speed * 0.4, limit_speed * 0.95)

            # 入彎技巧 (Entry Ratio)
            # 30% 機率練習「帶煞入彎」(Trail Braking, ratio > 1.0)
            if np.random.rand() < 0.3:
                entry_ratio = np.random.uniform(1.2, 1.8)
            else:
                entry_ratio = np.random.uniform(0.8, 1.1)

            # 柔順度 (Ramp)
            ramp = np.random.uniform(0.1, 0.4)

            return self.generate_turn(angle, radius, target_speed, entry_ratio, ramp)

        # --- C. Chicane (S型/閃避) ---
        elif motion_type == MotionType.chicane:
            # 長度：3m ~ 6m
            length = np.random.uniform(3.0, 6.0)
            # 擺盪寬度：0.5m (小閃避) ~ 1.5m (換車道)
            width = np.random.uniform(0.5, 1.5)
            # 速度：中高速
            speed = np.random.uniform(0.8, self.sim.cfg.max_linear_vel * 0.8)
            return self.generate_chicane(length, width, speed)

        # --- D. Threading (穿針引線) ---
        elif motion_type == MotionType.threading:
            # 長度：不用太長，專注於精準
            length = np.random.uniform(2.0, 4.0)
            # 偏移：模擬沒對準中心線 (-0.2m ~ 0.2m)
            offset = np.random.uniform(-0.2, 0.2)
            # 速度：慢！ (0.2 ~ 0.6)
            speed = np.random.uniform(0.2, 0.6)
            return self.generate_threading(length, offset, speed)

        # --- E. Brake (急煞) ---
        elif motion_type == MotionType.brake:
            # 初速：必須夠快才有煞車的意義
            init_speed = np.random.uniform(1.0, self.sim.cfg.max_linear_vel)
            # 反應時間：0.2s ~ 0.8s (模擬感測器延遲或人類延遲)
            reaction = np.random.uniform(0.2, 0.8)
            return self.generate_brake(init_speed, reaction)

        # --- F. Spin (原地轉) ---
        elif motion_type == MotionType.spin:
            direction = 1 if np.random.rand() > 0.5 else -1
            # 角度：90度 (L型死路) ~ 180度 (掉頭)
            angle = np.deg2rad(np.random.uniform(90, 180)) * direction
            # 轉速：中偏快
            rate = np.random.uniform(1.0, self.sim.cfg.max_angular_vel)
            return self.generate_spin(angle, rate)

        # --- G. Backward (倒車) ---
        elif motion_type == MotionType.backward:
            # 1. 隨機決定幾何參數
            dist = np.random.uniform(1.0, 3.0)      # 距離
            speed = np.random.uniform(0.2, 0.5)     # 速度 (偏慢)

            # 2. 決定是「直線」還是「轉彎」
            # 50% 機率直線，50% 機率轉彎
            if np.random.rand() > 0.5:
                # 直線模式：曲率接近 0
                # 加入極微小的雜訊 (-0.05 ~ 0.05 rad/m)，模擬人無法倒得筆直
                curvature = np.random.uniform(-0.05, 0.05)
            else:
                # 轉彎模式：曲率 = 1 / R
                radius = np.random.uniform(1.0, 3.0)    # 半徑
                direction = 1 if np.random.rand() > 0.5 else -1 # 左或右
                curvature = (1.0 / radius) * direction

            # 3. 呼叫純淨的執行函數
            return self.generate_backward(dist, speed, curvature)

        else:
            raise ValueError(f"Unsupported MotionType: {motion_type}")

    def generate_forward(self, total_time: float, speed: float) -> Trajectory:
        states = []
        commands = []
        time_elapsed = 0.0
        dt = self.sim.cfg.dt

        self.sim.reset()
        if np.random.rand() < 0.8:
            self.sim.actual_linear_vel = speed + np.random.uniform(-0.2, 0.2)
        else:
            self.sim.actual_linear_vel = 0.0

        if np.random.rand() < 0.5:
            self.sim.actual_angular_vel = np.random.uniform(-0.2, 0.2)
        else:
            self.sim.actual_angular_vel = 0.0

        while time_elapsed < total_time:
            pose, actual_vel = self.sim.step(speed, 0.0)
            states.append([pose[0], pose[1], pose[2]])
            commands.append([actual_vel[0], actual_vel[1]])
            time_elapsed += dt

        return Trajectory(states, commands, MotionType.forward, dt)

    def generate_turn(self,
                      turn_angle: float,
                      turn_radius: float,
                      target_speed: float,    # 彎中目標速度 (m/s)
                      entry_speed_ratio: float = 1.0, # 入彎初速比率 (1.0=完美, >1.0=帶煞入彎)
                      ramp_ratio: float = 0.2 # 梯形緩衝比例 (0.0~0.5), 控制動作柔順度
                      ) -> Trajectory:

        dt = self.sim.cfg.dt

        # 1. 計算幾何與動力學參數
        # 絕對值的弧長
        arc_length = turn_radius * abs(turn_angle)

        # 基礎持續時間 (如果完全不減速)
        base_duration = arc_length / max(target_speed, 0.1)

        # 修正持續時間：因為有加減速(Ramp)，平均角速度較低，所以總時間要拉長一點才能轉完角度
        # 經驗公式：時間拉長 (1 + ramp_ratio * 0.5)
        action_duration = base_duration * (1.0 + ramp_ratio * 0.5)

        # 計算最高角速度 (Peak Omega): w = v / R
        # 加上正負號決定方向
        peak_w = (target_speed / turn_radius) * np.sign(turn_angle)

        # 2. Initial State
        self.sim.reset()
        self.sim.actual_linear_vel = target_speed * entry_speed_ratio
        self.sim.actual_linear_vel = np.clip(self.sim.actual_linear_vel, 0.0, self.sim.cfg.max_linear_vel)

        # 3. 定義時間軸 (Timeline)
        t_run_in = 1.5   # 助跑段 (建立慣性)
        t_run_out = 1.5  # 收尾段 (建立 Local Goal)
        total_time = t_run_in + action_duration + t_run_out

        # 定義梯形的時間點
        t_ramp_up_end = t_run_in + (action_duration * ramp_ratio)
        t_ramp_down_start = t_run_in + action_duration * (1.0 - ramp_ratio)
        t_action_end = t_run_in + action_duration

        states = []
        commands = []
        time_elapsed = 0.0

        while time_elapsed < total_time:
            t = time_elapsed

            cmd_v = target_speed
            cmd_w = 0.0

            if t < t_run_in:
                # [Phase 0: 助跑] 直行
                cmd_w = 0.0

            elif t < t_ramp_up_end:
                # [Phase 1: 入彎] 角速度線性增加 (Linear Ramp Up)
                # 比例：0.0 -> 1.0
                progress = (t - t_run_in) / (action_duration * ramp_ratio)
                cmd_w = peak_w * progress

            elif t < t_ramp_down_start:
                # [Phase 2: 彎中] 保持最大角速度 (Steady Turn)
                cmd_w = peak_w

            elif t < t_action_end:
                # [Phase 3: 出彎] 角速度線性減少 (Linear Ramp Down)
                # 比例：1.0 -> 0.0
                time_left = t_action_end - t
                progress = time_left / (action_duration * ramp_ratio)
                cmd_w = peak_w * progress

            else:
                # [Phase 4: 收尾] 回正直行
                cmd_w = 0.0

            pose, actual_vel = self.sim.step(cmd_v, cmd_w)

            states.append(pose.tolist())
            commands.append(actual_vel.tolist())
            time_elapsed += dt

        return Trajectory(states, commands, MotionType.turn, dt)

    def generate_chicane(self,
                         segment_length: float, # S-turn 的總前進距離
                         width: float,          # 左右擺盪的幅度 (大約值)
                         speed: float           # 前進速度
                         ) -> Trajectory:

        dt = self.sim.cfg.dt
        duration = segment_length / max(speed, 0.1)

        # S型特徵：完成一個完整的正弦波週期 (0 -> 2pi)
        # 這樣車頭方向最後會回正，適合繼續接直線
        # 估算 w 的振幅: 根據擺盪寬度反推 (經驗公式)
        # w_peak 越大，擺盪越寬
        w_peak = (width / segment_length) * 4.0
        w_peak = np.clip(w_peak, 0.1, self.sim.cfg.max_angular_vel)

        # 隨機決定先左還先右
        direction = 1 if np.random.rand() > 0.5 else -1

        # 設定初始狀態
        self.sim.reset()
        self.sim.actual_linear_vel = speed
        self.sim.actual_angular_vel = 0.0

        # 時間軸
        t_run_in = 1.5
        t_run_out = 1.5
        total_time = t_run_in + duration + t_run_out

        states, commands = [], []
        time_elapsed = 0.0

        while time_elapsed < total_time:
            t = time_elapsed
            cmd_v = speed
            cmd_w = 0.0

            if t_run_in <= t < (t_run_in + duration):
                # [Action] 執行正弦波轉向
                # map time to [0, 2pi]
                phase = ((t - t_run_in) / duration) * 2 * np.pi
                cmd_w = direction * w_peak * np.sin(phase)

            # 執行
            pose, vel = self.sim.step(cmd_v, cmd_w)
            states.append(pose.tolist())
            commands.append(vel.tolist())
            time_elapsed += dt

        return Trajectory(states, commands, MotionType.chicane, dt)

    def generate_threading(self,
                           distance: float,
                           lateral_offset: float, # 目標稍微偏左或偏右 (模擬對不準)
                           speed: float           # 通常很低 (e.g. 0.3 ~ 0.5)
                           ) -> Trajectory:
        dt = self.sim.cfg.dt
        duration = distance / max(speed, 0.1)

        # Threading 的關鍵是 "慢" 且 "穩"
        # 這裡我們模擬一個簡單的 P-Controller 試圖修正 lateral_offset
        # 或者簡單一點：給一個極小的恆定 w，模擬機器人稍微走歪
        drift_w = (lateral_offset / distance) * 0.5
        drift_w = np.clip(drift_w, -0.2, 0.2)

        # 初始狀態：可能帶有一點點角度偏差
        init_theta_noise = np.random.uniform(-0.05, 0.05)
        self.sim.reset()
        self.sim.actual_linear_vel = speed
        self.sim.actual_angular_vel = 0.0
        self.sim.theta = init_theta_noise # 注入初始角度誤差

        t_run_in = 1.0  # Threading 助跑短一點沒關係
        t_run_out = 1.0
        total_time = t_run_in + duration + t_run_out

        states, commands = [], []
        time_elapsed = 0.0

        while time_elapsed < total_time:
            t = time_elapsed
            cmd_v = speed
            cmd_w = 0.0

            if t_run_in <= t < (t_run_in + duration):
                # [Action]
                # 疊加高頻雜訊 (模擬在窄縫中輪子打滑或顛簸)
                noise = np.random.normal(0, 0.05)
                cmd_w = drift_w + noise

            pose, vel = self.sim.step(cmd_v, cmd_w)
            states.append(pose.tolist())
            commands.append(vel.tolist())
            time_elapsed += dt

        return Trajectory(states, commands, MotionType.threading, dt)

    def generate_brake(self,
                       initial_speed: float,
                       reaction_time: float = 0.5 # 煞車前的反應時間
                       ) -> Trajectory:
        dt = self.sim.cfg.dt

        # 估算停下來需要的時間 v = a*t -> t = v/a
        stop_time_est = initial_speed / self.sim.cfg.max_linear_acc

        # 總時長
        t_run_in = 1.5      # 穩定巡航
        t_braking = stop_time_est + 1.0 # 煞車過程 + 緩衝
        t_steady_stop = 1.5 # 完全停穩後的一段時間 (很重要! 讓模型學會 v=0 是合法的)

        total_time = t_run_in + reaction_time + t_braking + t_steady_stop

        # 初始狀態：全速前進
        self.sim.reset()
        self.sim.actual_linear_vel = initial_speed
        self.sim.actual_angular_vel = 0.0

        states, commands = [], []
        time_elapsed = 0.0

        while time_elapsed < total_time:
            t = time_elapsed

            if t < (t_run_in + reaction_time):
                # [Phase 1: 沒看到障礙物前] 保持全速
                cmd_v = initial_speed
                cmd_w = 0.0
            else:
                # [Phase 2: 看到障礙物/急煞] 指令歸零
                # Simulator 會根據 ACC_LIM 慢慢把 actual_v 降下來
                cmd_v = 0.0
                cmd_w = 0.0

            pose, vel = self.sim.step(cmd_v, cmd_w)
            states.append(pose.tolist())
            commands.append(vel.tolist())
            time_elapsed += dt

        return Trajectory(states, commands, MotionType.brake, dt)

    def generate_spin(self,
                      spin_angle: float,      # 目標總角度 (rad)
                      spin_rate: float,       # 最大角速度 (rad/s)
                      ramp_ratio: float = 0.2 # 加減速佔比 (0.0~1.0)
                      ) -> Trajectory:
        """
        生成原地旋轉軌跡。
        修正了時間計算，確保旋轉角度精準。
        """
        dt = self.sim.cfg.dt

        # 1. 參數準備
        abs_angle = abs(spin_angle)
        peak_w = abs(spin_rate) * np.sign(spin_angle)

        # 2. 精確計算持續時間 (Correction Here!)
        # 使用梯形面積公式反推： T = Angle / (Peak_W * (1 - 0.5 * ramp_ratio))
        # 避免分母為 0
        safe_ramp = np.clip(ramp_ratio, 0.0, 0.99)
        action_duration = abs_angle / (abs(spin_rate) * (1.0 - 0.5 * safe_ramp))

        # 3. 初始狀態
        self.sim.reset()
        self.sim.actual_linear_vel = 0.0
        self.sim.actual_angular_vel = 0.0

        t_run_in = 1.0  # 靜止等待
        t_run_out = 1.0 # 轉完後靜止
        total_time = t_run_in + action_duration + t_run_out

        # 計算 ramp 的時間點
        # 總 ramp 時間 = duration * ramp_ratio
        # 單邊 ramp 時間 = 總 ramp 時間 / 2 (假設對稱)
        t_ramp_duration = action_duration * safe_ramp / 2.0

        t_up_end = t_run_in + t_ramp_duration
        t_down_start = t_run_in + action_duration - t_ramp_duration
        t_action_end = t_run_in + action_duration

        states, commands = [], []
        time_elapsed = 0.0

        while time_elapsed < total_time:
            t = time_elapsed
            cmd_v = 0.0 # Spin 時線速度恆為 0
            cmd_w = 0.0

            if t < t_run_in:
                # 助跑段：靜止
                cmd_w = 0.0

            elif t < t_action_end:
                # 動作段
                if t < t_up_end:
                    # [Ramp Up] 線性加速 0 -> peak
                    progress = (t - t_run_in) / t_ramp_duration
                    cmd_w = peak_w * progress

                elif t > t_down_start:
                    # [Ramp Down] 線性減速 peak -> 0
                    time_left = t_action_end - t
                    progress = time_left / t_ramp_duration
                    cmd_w = peak_w * progress

                else:
                    # [Steady] 保持最大轉速
                    cmd_w = peak_w

            else:
                # 收尾段：靜止
                cmd_w = 0.0

            pose, vel = self.sim.step(cmd_v, cmd_w)
            states.append(pose.tolist())
            commands.append(vel.tolist())
            time_elapsed += dt

        return Trajectory(states, commands, MotionType.spin, dt)

    def generate_backward(self,
                          distance: float,
                          speed: float,
                          curvature: float = 0.0 # 預設為 0 (直線)
                          ) -> Trajectory:
        """
        生成倒車軌跡 (包含直線或轉彎)。

        :param distance: 倒車總距離 (m)
        :param speed: 倒車速度 (m/s)，函數內會自動轉為負值
        :param curvature: 軌跡曲率 (1/m)。
                          0.0 = 直線
                          正值 = 車尾向左轉 (逆時針軌跡)
                          負值 = 車尾向右轉 (順時針軌跡)
        """
        dt = self.sim.cfg.dt

        # 1. 物理指令計算
        target_v = -abs(speed) # 確保是倒車

        # 運動學公式: w = v * k (角速度 = 線速度 * 曲率)
        # 倒車時 v 為負，若 curvature 為正，w 也會變負 -> 符合物理 (倒車入庫方向盤打左，車身順時針轉)
        target_w = target_v * curvature

        # 2. 時間計算
        # 倒車距離 / 速度
        duration = distance / max(abs(speed), 0.1)

        # 3. 初始狀態
        self.sim.reset()
        self.sim.actual_linear_vel = 0.0
        self.sim.actual_angular_vel = 0.0

        t_run_in = 1.0  # 靜止思考時間
        t_run_out = 1.0 # 停穩
        total_time = t_run_in + duration + t_run_out

        states, commands = [], []
        time_elapsed = 0.0

        while time_elapsed < total_time:
            t = time_elapsed

            # 簡單的開關控制 (Bang-bang control for simplicity in backward)
            # 也可以改成梯形，但倒車通常動作比較慢且直接
            if t_run_in <= t < (t_run_in + duration):
                cmd_v = target_v
                cmd_w = target_w
            else:
                cmd_v = 0.0
                cmd_w = 0.0

            pose, vel = self.sim.step(cmd_v, cmd_w)
            states.append(pose.tolist())
            commands.append(vel.tolist())
            time_elapsed += dt

        return Trajectory(states, commands, MotionType.backward, dt)
