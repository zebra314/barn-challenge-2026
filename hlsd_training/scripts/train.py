import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time

from models import HLSDLite

# --- 訓練超參數 ---
class TrainConfig:
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    SMOOTHNESS_WEIGHT = 0.5  # 平滑項權重 (越高越穩，但反應越慢)
    VAL_SPLIT = 0.2          # 驗證集比例

class BARNDataset(Dataset):
    def __init__(self, npz_file):
        print(f"Loading dataset from {npz_file}...")
        data = np.load(npz_file)

        # 轉為 PyTorch Tensor
        self.X = torch.from_numpy(data['X']).float()
        self.Y = torch.from_numpy(data['Y']).float()

        # 檢查數據是否有 NaN
        if torch.isnan(self.X).any() or torch.isnan(self.Y).any():
            print("[Warning] Dataset contains NaN! Replacing with 0.")
            self.X = torch.nan_to_num(self.X)
            self.Y = torch.nan_to_num(self.Y)

        print(f"Dataset Ready. Input: {self.X.shape}, Label: {self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train(args):
    # 1. 設定路徑與裝置
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"[Error] Data file not found: {data_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mac M1/M2 加速支援
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Training on device: {device}")

    # 2. 準備數據
    full_dataset = BARNDataset(data_path)

    # 切分訓練/驗證集
    val_size = int(len(full_dataset) * TrainConfig.VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. 初始化模型與優化器
    model = HLSDLite(input_size=64, output_size=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TrainConfig.LEARNING_RATE)
    criterion = nn.MSELoss()

    # 4. 訓練迴圈
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(TrainConfig.EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # --- Loss Calculation ---
            # 1. Task Loss (預測誤差)
            task_loss = criterion(outputs, targets)

            # 2. Smoothness Loss (平滑項)
            # Input format: [Lidar(60), Goal(2), Last_Vel(2)]
            # 提取輸入中的 "上一幀速度" (Last_Vel)
            last_vel_inputs = inputs[:, -2:]

            # 這裡要注意: Input 的 Last_Vel 是歸一化過的 (0~1)，
            # 但 Model Output 是預測的實際速度 (例如 0~2.0)。
            # 為了正確比較，我們需要知道歸一化參數。
            # 簡單做法：假設訓練數據生成時 Label 是 Raw Value，而 Input Last Vel 是 Normalized。
            # *修正策略*: 在 02_hallucinate.py 中，我們把 input 的 last_vel 除以了 MAX_V。
            # 所以這裡要把 input 的 last_vel 還原回去，才能跟 output 比較。

            # 簡單還原 (假設 max_v=2.0, max_w=2.0，這要跟 Config 一致)
            # 這裡寫死係數，建議未來從 Config 讀入
            restored_last_v = last_vel_inputs[:, 0] * 2.0
            restored_last_w = last_vel_inputs[:, 1] * 2.0
            restored_last_vel = torch.stack([restored_last_v, restored_last_w], dim=1)

            smooth_loss = criterion(outputs, restored_last_vel)

            # Total Loss
            loss = task_loss + TrainConfig.SMOOTHNESS_WEIGHT * smooth_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Validation Loss (只看 MSE，平滑項主要用於約束訓練)
                # 當然也可以加上平滑項看總體表現
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{TrainConfig.EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

    print(f"Training finished in {time.time() - start_time:.1f}s")

    # 5. 保存模型
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # A. 保存 PyTorch 權重 (方便以後繼續訓練)
    pth_path = out_dir / "hlsd_model.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"[Save] PyTorch model saved to {pth_path}")

    # B. 導出 ONNX (給 C++ Plugin 用)
    onnx_path = out_dir / "hlsd_model.onnx"
    model.export_to_onnx(str(onnx_path))

    # 6. 繪製並保存曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss (MSE Only)')
    plt.title('HLSD-Lite Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "loss_curve.png")
    print(f"[Save] Loss curve saved to {out_dir / 'loss_curve.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 自動尋找 data 路徑
    default_data_path = Path(__file__).resolve().parent.parent / 'data/training_dataset.npz'

    parser.add_argument('--data_file', type=str, default=str(default_data_path),
                        help='Path to .npz dataset')
    parser.add_argument('--out_dir', type=str, default='models',
                        help='Output directory for models')

    args = parser.parse_args()

    train(args)
