import torch
from tensordict import TensorDict


def print_tensor(tensor: torch.Tensor, extra_str: str = None):
    # print(tensor.shape)
    # print(tensor.dtype)
    # print(tensor.device)
    if tensor.dim() >= 2:
        tensor = tensor[0]
    nums = tensor.tolist()
    nums = [str( num) for num in nums]
    str_to_print = '\n' +'\t'.join(nums)
    print('--------------')
    if extra_str:
        print(extra_str)
    print(str_to_print)
    print('--------------')

def print_tensor_dict(tensor_dict: TensorDict, extra_str: str = None, keys: list = None):
    if extra_str:
        print(extra_str)
    if keys:
        for key in keys:
            print_tensor(tensor_dict[key], key)
        return
    for key, value in tensor_dict.items():
        print_tensor(value, key)