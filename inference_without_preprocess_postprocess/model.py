import torch
import torch.nn as nn

# === 1. 构建简单线性模型并固定权重 ===
model = nn.Linear(128, 4)
with torch.no_grad():
    model.weight.fill_(1.0)   # 权重全为 1.0
    model.bias.fill_(128.0)

model.eval()  # 切换到评估模式

# === 2. 创建测试输入（全1，float32，形状 1x128）===
x = torch.ones(1, 128, dtype=torch.float32)

# === 3. 测试前向结果 ===
with torch.no_grad():
    y = model(x)
print("Test output:", y)  # 应输出 tensor([[128., 128., 128., 128.]])

# === 4. 直接导出 ONNX ===
torch.onnx.export(
    model,
    x,
    "linear_128_to_4.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print("✅ ONNX 模型已导出为 'linear_128_to_4.onnx'")