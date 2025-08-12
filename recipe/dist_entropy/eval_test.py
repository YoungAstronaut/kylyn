import torch
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# 检查是否有可用的 GPU
if __name__ == "__main__":
    gpu = sys.argv[1]
    if torch.cuda.is_available():
        print(f"using GPU {gpu}")
        device = torch.device(f"cuda:{gpu}")  # 使用 GPU
    else:
        device = torch.device("cpu")   # 如果没有 GPU，使用 CPU

    # 创建一个张量，并将其移动到 GPU
    tensor = torch.rand(60000, 60000).to(device)

    # 创建一个简单的计算任务，可以根据需要调整
    result = tensor.mm(tensor)

    # 一直运行以占用 GPU
    while True:
        result = tensor.mm(tensor)

    # 添加更多的计算任务以更充分地占用 GPU