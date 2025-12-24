import torch
import torch.nn as nn
import torch.nn.init as init

class HLSDLite(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=2):
        super(HLSDLite, self).__init__()

        # 輕量級 MLP 架構
        # Input (64) -> FC (128) -> ReLU -> FC (64) -> ReLU -> Output (2)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

        # 權重初始化 (Kaiming He Initialization)
        # 這對 ReLU 網絡很重要，防止訓練初期神經元死亡
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

    def export_to_onnx(self, file_path="hlsd_model.onnx", input_sample=None):
        """
        將模型導出為 ONNX 格式 (用於 OpenVINO/C++ 推理)
        """
        self.eval() # 切換到評估模式

        if input_sample is None:
            # 建立一個假的輸入數據 (Batch=1, Input=64)
            input_sample = torch.randn(1, 64)

        torch.onnx.export(
            self,
            input_sample,
            file_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Model exported to {file_path}")

if __name__ == "__main__":
    # 簡單測試
    model = HLSDLite()
    print(model)
    dummy_input = torch.randn(5, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 應該是 [5, 2]
