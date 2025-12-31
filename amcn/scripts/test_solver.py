import time
import yaml
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from amcn.optimization.mpc_solver import CasadiMPC

def get_reference_trajectory(t_start, dt, N, speed=1.0):
    """
    生成一條測試用的參考軌跡 (這裡是正弦波)
    Return shape: (3, N+1) -> [x, y, theta]
    """
    ref_traj = np.zeros((3, N + 1))

    for k in range(N + 1):
        t = t_start + k * dt

        # 定義路徑: x = t, y = sin(0.5 * t)
        x = speed * t
        y = np.sin(0.5 * x)

        # 計算切線角度 theta = atan2(dy, dx)
        dx = speed
        dy = 0.5 * speed * np.cos(0.5 * x)
        theta = np.arctan2(dy, dx)

        ref_traj[0, k] = x
        ref_traj[1, k] = y
        ref_traj[2, k] = theta

    return ref_traj

def simulate_robot(current_state, v, omega, dt):
    """
    簡單的運動學模擬器 (用來更新機器人位置)
    x_{k+1} = x_k + v * cos(theta) * dt
    """
    x, y, theta = current_state

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt

    return np.array([x_new, y_new, theta_new])

def main():
    config_path = Path(__file__).parent.parent / 'configs' / 'mpc_params.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Initializing MPC Solver...")
    config = config['mpc']
    mpc = CasadiMPC(config)

    # 2. 初始化模擬狀態
    current_state = np.array([0.0, 0.0, 0.5]) # 起點 [x, y, theta] (故意設一個歪的角度測試修正能力)
    sim_time = 20.0 # 模擬總時間 (秒)
    total_steps = int(sim_time / config['dt'])

    # 紀錄數據用於繪圖
    history_x = []
    history_y = []
    history_v = []
    history_w = []
    ref_x = []
    ref_y = []

    print(f"Starting simulation for {sim_time} seconds...")
    start_real_time = time.time()

    # 3. 主循環
    for i in range(total_steps):
        t_now = i * config['dt']

        # A. 獲取局部參考軌跡 (Local Horizon)
        # 在真實 ROS 中，這來自全域路徑切片；這裡數學生成
        ref_traj = get_reference_trajectory(t_now, config['dt'], config['horizon'])

        # B. 執行 MPC 求解
        try:
            v, omega = mpc.solve(current_state, ref_traj)
        except Exception as e:
            print(f"Solver failed at step {i}: {e}")
            break

        # C. 模擬機器人運動 (Apply Control)
        current_state = simulate_robot(current_state, v, omega, config['dt'])

        # D. 紀錄數據
        history_x.append(current_state[0])
        history_y.append(current_state[1])
        history_v.append(v)
        history_w.append(omega)
        ref_x.append(ref_traj[0, 0]) # 只紀錄參考軌跡的第一個點用於對比
        ref_y.append(ref_traj[1, 0])

        # 簡單的進度條
        if i % 10 == 0:
            sys.stdout.write(f"\rStep {i}/{total_steps} | Pos: ({current_state[0]:.2f}, {current_state[1]:.2f}) | Vel: {v:.2f}")
            sys.stdout.flush()

    print(f"\nSimulation done. Computation FPS: {total_steps / (time.time() - start_real_time):.2f}")

    # 4. 繪圖驗證
    plt.figure(figsize=(12, 8))

    # 子圖 1: 軌跡追蹤
    plt.subplot(2, 1, 1)
    plt.plot(ref_x, ref_y, 'r--', label='Reference Path (Sine)')
    plt.plot(history_x, history_y, 'b-', linewidth=2, label='MPC Trajectory')
    plt.title('Phase 1 MPC Tracking Performance')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 子圖 2: 控制輸入
    plt.subplot(2, 1, 2)
    plt.plot(history_v, 'g-', label='Linear Velocity (v)')
    plt.plot(history_w, 'y-', label='Angular Velocity (omega)')
    plt.axhline(y=config['v_max'], color='r', linestyle=':', label='Max V')
    plt.axhline(y=-config['omega_max'], color='orange', linestyle=':', label='Min Omega')
    plt.axhline(y=config['omega_max'], color='orange', linestyle=':', label='Max Omega')
    plt.title('Control Inputs')
    plt.xlabel('Step')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
