import torch

def print_tensor(tensor: torch.Tensor):
    print(tensor.shape)
    print(tensor.dtype)
    print(tensor.device)
    if tensor.dim() >= 2:
        tensor = tensor[0]
    nums = tensor.tolist()
    nums = [str( num) for num in nums]
    str_to_print = ' | '.join(nums)
    print('--------------')
    print(str_to_print)
    print('--------------')