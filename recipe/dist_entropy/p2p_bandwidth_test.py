import torch
import torch.distributed as dist
import time

# 初始化进程组（限定2个GPU）
dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

# 测试参数
data_size = 128 * 1024**2  # 128MB
tensor = torch.ones(data_size // 4, dtype=torch.float32).cuda()  # 单精度浮点占4字节

# 定义发送方(rank0)和接收方(rank1)
if rank == 0:
    dist.send(tensor, dst=1)  # GPU0发送数据到GPU1
elif rank == 1:
    dist.recv(tensor, src=0)  # GPU1接收来自GPU0的数据

# 同步计时
dist.barrier()
start_time = time.time()

# 执行10次传输取平均值（减少误差）
for _ in range(10):
    if rank == 0:
        dist.send(tensor, dst=1)
    elif rank == 1:
        dist.recv(tensor, src=0)

# 计算带宽
dist.barrier()
elapsed = time.time() - start_time
total_data = data_size * 10  # 10次传输总量
bandwidth = total_data / (elapsed * 1024**3)  # 单位GB/s

if rank == 0:
    print(f"GPU0 → GPU1 带宽: {bandwidth:.2f} GB/s (数据量: {data_size//1024**2}MB x 10次)")
elif rank == 1:
    print(f"GPU1 → GPU0 带宽: {bandwidth:.2f} GB/s (数据量: {data_size//1024**2}MB x 10次)")