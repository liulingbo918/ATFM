import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from model.spn import ModelAttentionWithTimeaware as Model
from dataset.dataset import DatasetFactory

class DataConfiguration:
    # Data
    name = 'BikeNYC'
    portion = 1.  # portion of data

    len_close = 4
    len_period = 2
    len_trend = 0
    pad_forward_period = 0
    pad_back_period = 0
    pad_forward_trend = 0
    pad_back_trend = 0

    len_all_close = len_close * 1
    len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
    len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)

    len_seq = len_all_close + len_all_period + len_all_trend
    cpt = [len_all_close, len_all_period, len_all_trend]

    interval_period = 1
    interval_trend = 7

    ext_flag = True
    ext_time_flag = True
    rm_incomplete_flag = True
    fourty_eight = False
    previous_meteorol = True

    ext_dim = 33
    dim_h = 16
    dim_w = 8

def test(dconf):
    np.random.seed(777)
    torch.manual_seed(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ds_factory = DatasetFactory(dconf)
    test_ds = ds_factory.get_test_dataset()

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=1
    )

    model = Model(dconf)
    try:
        model.load_state_dict(torch.load('pretrained/BikeNYC/model'))
    except:
        model = torch.load('pretrained/BikeNYC/model')
    model = model.cuda()

    criterion = nn.MSELoss().cuda()

    model.eval()
    mse = 0.0
    mse_in = 0.0
    mse_out = 0.0
    mmn = ds_factory.ds.mmn
    with torch.no_grad():
        for i, (X, X_ext, Y, Y_ext) in enumerate(test_loader, 0):
            X = X.cuda()
            X_ext = X_ext.cuda() 
            Y = Y.cuda() 
            Y_ext = Y_ext.cuda()
            h = model(X, X_ext, Y_ext)
            loss = criterion(h, Y)
            mse += X.size()[0] * loss.item()

            mse_in += X.size()[0] * torch.mean((Y[:, 0] - h[:, 0]) * (Y[:, 0] - h[:, 0])).item()
            mse_out += X.size()[0] * torch.mean((Y[:, 1] - h[:, 1]) * (Y[:, 1] - h[:, 1])).item()

    cnt = ds_factory.ds.X_test.shape[0]
    mse /= cnt
    rmse = math.sqrt(mse) * (mmn.max - mmn.min) / 2. * ds_factory.dataset.m_factor
    print("rmse: %.4f" % (rmse))

    mse_in /= cnt
    rmse_in = math.sqrt(mse_in) * (mmn.max - mmn.min) / 2. * ds_factory.dataset.m_factor
    mse_out /= cnt
    rmse_out = math.sqrt(mse_out) * (mmn.max - mmn.min) / 2. * ds_factory.dataset.m_factor

    print("inflow rmse: %.4f    outflow rmse: %.4f" % (rmse_in, rmse_out))

if __name__ == '__main__':
    dconf = DataConfiguration()

    test(dconf)