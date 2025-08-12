import torch
import torch.distributed as dist
import time

# 初始化NCCL
dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

# 测试参数
data_size = 128 * 1024**2  # 128MB
tensor = torch.ones(data_size // 4, dtype=torch.float32).cuda()  # 单精度浮点占4字节

# 同步所有GPU
dist.barrier()
start_time = time.time()

# 执行AllReduce操作（模拟梯度同步）
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# 计算带宽
dist.barrier()
elapsed = time.time() - start_time
bandwidth = (data_size * 2) / (elapsed * 1024**3)  # 双向流量：数据大小×2 → 单位GB/s

if rank == 0:
    print(f"带宽实测: {bandwidth:.2f} GB/s (数据量: {data_size//1024**2}MB)")