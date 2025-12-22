import numpy as np
from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

from simulation.jackal_simulator import JackalSimulator
from simulation.motion_generator import MotionGenerator
from hallucination.hallucinate_generator import Config, process_single_frame
from utils.logging_config import setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Generate motions ----------------------------- #
    sim = JackalSimulator()
    generator = MotionGenerator(sim)

    states, cmds = generator.generate_episode(total_episodes=10, steps_per_episode=1000)
    np.save(data_dir / 'raw_episodes_states.npy', states)
    np.save(data_dir / 'raw_episodes_commands.npy', cmds)
    logger.info(f"Generated raw motion data and saved to {data_dir}")

    # -------------------------- Generate Hallucination -------------------------- #

    # input_states_file = data_dir / 'raw_states.npy'
    # input_cmds_file = data_dir / 'raw_commands.npy'
    # output_dataset_file = data_dir / 'training_dataset.npz'

    # # 2. 設定運算資源
    # # 留 1~2 個核心給系統，避免電腦死當
    # num_workers = max(1, cpu_count() - 2)

    # print(f"=== BARN Hallucination Dataset Generator ===")
    # print(f"Input Directory:  {data_dir}")
    # print(f"Output File:      {output_dataset_file}")
    # print(f"CPU Workers:      {num_workers}")
    # print("============================================")

    # # ================= 1. 載入原始數據 =================
    # if not input_states_file.exists():
    #     print(f"[Error] Not found: {input_states_file}")
    #     print("Please run '01_gen_motion.py' first to generate raw trajectories.")
    #     return

    # print(f"[1/4] Loading raw data...")
    # t_start = time.time()

    # # 這裡讀入的是包含多條軌跡的大陣列 (中間有 NaN 隔開)
    # raw_states = np.load(input_states_file)     # shape: [Total_Steps, 5]
    # raw_commands = np.load(input_cmds_file)     # shape: [Total_Steps, 2]

    # print(f"      Loaded {len(raw_states)} raw frames.")
    # print(f"      Time elapsed: {time.time() - t_start:.2f}s")

    # # ================= 2. 準備任務 (Task Dispatching) =================
    # print(f"[2/4] Preparing hallucination tasks...")

    # horizon_steps = int(Config.HORIZON_SEC / Config.DT)
    # tasks = []

    # # 我們需要遍歷所有數據，找出「合法的視窗」
    # # 合法視窗定義：從 t 到 t+horizon 的這段時間內，數據都不能是 NaN
    # # (如果遇到 NaN，代表跨越了兩次獨立的模擬 Episode)

    # valid_range = range(1, len(raw_states) - horizon_steps)

    # # 預先檢查 NaN 以加速篩選 (這比在迴圈裡檢查快)
    # is_nan = np.isnan(raw_states).any(axis=1)

    # for i in tqdm(valid_range, desc="Scanning valid windows"):
    #     # 檢查視窗頭尾是否有 NaN (通常這樣就夠了，因為 NaN 是連續的一大塊)
    #     if is_nan[i] or is_nan[i + horizon_steps]:
    #         continue

    #     # 準備單一任務的數據包
    #     # 注意：我們在這裡就切片 (Slicing)，這樣傳進去 worker 的數據量才小
    #     curr_state = raw_states[i]
    #     future_states = raw_states[i : i + horizon_steps]

    #     # Input 需要上一幀的指令 (模擬延遲/慣性參考)
    #     last_cmd = raw_commands[i-1]

    #     # Label 是當下的理想指令
    #     curr_label = raw_commands[i]

    #     # 打包任務
    #     # 格式: (index, current_state, future_states, last_cmd_vel, label_cmd)
    #     tasks.append((i, curr_state, future_states, last_cmd, curr_label))

    # print(f"      Total valid training samples found: {len(tasks)}")

    # # ================= 3. 平行運算 (Multiprocessing) =================
    # print(f"[3/4] Running hallucination on {num_workers} cores...")
    # print("      This may take a while depending on dataset size...")

    # X_list = []
    # Y_list = []

    # # 使用 Imap + Tqdm 顯示即時進度
    # # chunksize 設定稍大一點可以減少進程切換開銷
    # with Pool(processes=num_workers) as pool:
    #     results = list(tqdm(pool.imap(process_single_frame, tasks, chunksize=100),
    #                        total=len(tasks),
    #                        desc="Hallucinating"))

    # # ================= 4. 整理與存檔 =================
    # print(f"[4/4] Stacking and saving data...")

    # # 過濾掉可能的 None 結果 (雖然理論上 task 篩選過應該不會有)
    # results = [r for r in results if r is not None]

    # if not results:
    #     print("[Error] No valid data generated!")
    #     return

    # # 解壓縮結果 list of tuples -> tuple of lists
    # X_list, Y_list = zip(*results)

    # X_final = np.array(X_list, dtype=np.float32)
    # Y_final = np.array(Y_list, dtype=np.float32)

    # print(f"      Final Dataset Shapes:")
    # print(f"      X (Input): {X_final.shape}  [Lidar(60) + Goal(2) + LastVel(2)]")
    # print(f"      Y (Label): {Y_final.shape}  [v, w]")

    # # 使用壓縮格式存檔，節省硬碟空間
    # np.savez_compressed(output_dataset_file, X=X_final, Y=Y_final)

    # print(f"\n[Success] Dataset saved to: {output_dataset_file}")
    # print("Next Step: Run '03_train.py' to train your HLSD model.")

if __name__ == "__main__":
    main()
