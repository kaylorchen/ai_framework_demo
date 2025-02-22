import torch
import torch.nn as nn

# 定义一个简单的神经网络
class Add10ToElements(nn.Module):
    def __init__(self):
        super(Add10ToElements, self).__init__()

    def forward(self, x):
        # 将输入的每个元素加 10
        return x + 10


# 导出模型为 ONNX 格式
def export_to_onnx(model, input_tensor, onnx_file_path):
    # 设置模型为评估模式
    model.eval()

    # 导出模型
    torch.onnx.export(
        model,  # 模型
        input_tensor,  # 输入张量
        onnx_file_path,  # 导出的 ONNX 文件路径
        opset_version=11,  # ONNX 算子集版本
        input_names=["input"],  # 输入名称
        output_names=["output"],  # 输出名称
    )
    print(f"模型已导出到: {onnx_file_path}")


# 测试网络并导出 ONNX
if __name__ == "__main__":
    # 初始化网络
    model = Add10ToElements()

    # 创建一个输入张量，形状为 [1, 3, 10, 10]，数据类型为 float32
    input_tensor = torch.randn(1, 3, 10, 10, dtype=torch.float32)
    print("输入张量的形状:", input_tensor.shape)
    print("输入张量的值（部分）:\n", input_tensor[0, :, :3, :3])

    # 前向传播
    output_tensor = model(input_tensor)
    print("输出张量的形状:", output_tensor.shape)
    print("输出张量的值（部分）:\n", output_tensor[0, :, :3, :3])

    # 导出模型为 ONNX 格式
    onnx_file_path = "add10_model.onnx"
    export_to_onnx(model, input_tensor, onnx_file_path)
