import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        # nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
        nn.init.xavier_normal_(model.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
        # nn.init.xavier_normal_(model.weight.data)

def RMSE(input, target, mmn, m_factor):
    rmse = torch.sqrt(F.mse_loss(input, target)) * (mmn.max - mmn.min) / 2. * m_factor
    return rmse

def MAPE(input, target, mmn, m_factor):
    mape = torch.mean(torch.abs((target - input) / input))
    return mape