import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Hallucination:
    """
    代表單一個時間點 (Frame) 的幻覺場景設定。
    包含機器人狀態、目標、以及周圍生成的障礙物列表。
    這個物件是 Ray Caster 的輸入。
    """
    frame_idx: int
    timestamp: float

    # --- 機器人狀態 ---
    # Global Frame [x, y, theta] (用於視覺化除錯)
    pose_global: np.ndarray
    # 當下實際速度 [v, w] (用於網路輸入 State)
    current_vel: np.ndarray

    # --- 訓練目標 (Labels) ---
    # 專家示範的指令 [v_target, w_target]
    target_cmd: np.ndarray

    # --- 導航目標 (Network Input) ---
    # Local Frame 中的目標點 [x_local, y_local]
    # 通常取 Future 軌跡的末端點
    local_goal: np.ndarray

    # --- 幻覺障礙物 ---
    # Global Frame 中的圓柱體列表
    # List of tuple: (center_x, center_y, radius)
    obstacles: List[Tuple[float, float, float]]
