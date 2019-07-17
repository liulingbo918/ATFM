import torch
import torch.nn as nn

def split_cpt(value, cpt):
    if not isinstance(value, torch.Tensor):
        raise ValueError('Parameter Value should be a Tensor.')
    scale = int(value.size()[1]) // sum(cpt)
    split_list = []
    for i in cpt:
        if i > 0:
            split_list.append(i * scale)
    if len(split_list) <= 0:
        raise ValueError('Get empty split_list.')
    
    return torch.split(value, split_size_or_sections=split_list, dim=1)

def split_seq(value, seq_len):
    if not isinstance(value, torch.Tensor):
        raise ValueError('Parameter Value should be a Tensor.')
    
    return torch.split(value, seq_len, dim=1)