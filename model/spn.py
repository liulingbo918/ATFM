import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer.resnn import ResNN
from model.layer.extnn import ExtNN
from model.layer.concat_conv import ConcatConv
from model.layer.conv_lstm import ConvLSTM

class ModelConfigurationTaxiBJ:
    """
    Change the class ModelConfiguration freely to fit different models' design
    """
    res_repetation = 12
    res_nbfilter = 16
    res_bn = True
    res_split_mode = 'split'  # 'split', 'split-chans', 'cpt', 'concat', 'none'

    first_extnn_inter_channels = 40
    first_extnn_dropout = 0.5

    merge_mode = 'fuse'  # 'LSTM', 'fuse'
    lstm_channels = 16
    lstm_dropout = 0

    extnn = True

class ModelConfigurationBikeNYC:
    res_repetation = 2
    res_nbfilter = 16
    res_bn = True
    res_split_mode = 'split'  # 'split', 'split-chans', 'cpt', 'concat', 'none'

    first_extnn_inter_channels = 30
    first_extnn_dropout = 0.5

    merge_mode = 'fuse'  # 'LSTM', 'fuse'
    lstm_channels = 16
    lstm_dropout = 0.5

    extnn = True

class ModelConfigurationTaxiNYC:
    """
    Change the class ModelConfiguration freely to fit different models' design
    """
    res_repetation = 4
    res_nbfilter = 32
    res_bn = True
    res_split_mode = 'split'  # 'split', 'split-chans', 'cpt', 'concat', 'none'

    first_extnn_inter_channels = 40
    first_extnn_dropout = 0.5

    merge_mode = 'fuse'  # 'LSTM', 'fuse'
    lstm_channels = 16
    lstm_dropout = 0

    extnn = True

class ModelAttentionWithTimeaware(nn.Module):
    def __init__(self, data_conf):
        super().__init__()

        # config
        self.dconf = data_conf
        if self.dconf.name == 'BikeNYC':
            self.mconf = ModelConfigurationBikeNYC()
        elif self.dconf.name == 'TaxiNYC':
            self.mconf = ModelConfigurationTaxiNYC()
        else:
            self.mconf = ModelConfigurationTaxiBJ()

        self.resnn = ResNN(
            in_channels=2*self.dconf.len_seq, 
            out_channels=self.mconf.lstm_channels*self.dconf.len_seq,
            inter_channels=self.mconf.res_nbfilter,
            repetation=self.mconf.res_repetation,
            bnmode=self.mconf.res_bn,
            splitmode=self.mconf.res_split_mode,
            cpt=self.dconf.cpt
        )
        self.extnn = ExtNN(
            in_features=self.dconf.ext_dim,
            out_height=self.dconf.dim_h,
            out_width=self.dconf.dim_w,
            out_channels=self.mconf.lstm_channels,
            inter_features=self.mconf.first_extnn_inter_channels,
            mode='inter',
            dropout_rate=self.mconf.first_extnn_dropout
        )

        self.len_local = self.dconf.len_all_close
        self.len_global = self.dconf.len_all_period + self.dconf.len_all_trend

        self.conv_lstm = ConvLSTM(
            in_channels=self.mconf.lstm_channels,
            height=self.dconf.dim_h,
            width=self.dconf.dim_w,
            lstm_channels=self.mconf.lstm_channels,
            all_hidden=True,
            mode='cpt',
            cpt=[self.len_local, self.len_global, 0],
            dropout_rate=self.mconf.lstm_dropout,
            last_conv=False
        )

        self.concat_conv_c = ConcatConv(
            in_channels1=2*self.len_local, 
            in_channels2=self.mconf.lstm_channels*self.len_local,
            out_channels=self.mconf.lstm_channels*self.len_local,
            inter_channels=self.mconf.lstm_channels,
            relu_conv=True,
            seq_len=self.len_local
        )
        self.concat_conv_t = ConcatConv(
            in_channels1=2*self.len_global, 
            in_channels2=self.mconf.lstm_channels*self.len_global,
            out_channels=self.mconf.lstm_channels*self.len_global,
            inter_channels=self.mconf.lstm_channels,
            relu_conv=True,
            seq_len=self.len_global
        )

        self.conv_lstm_c = ConvLSTM(
            in_channels=self.mconf.lstm_channels,
            height=self.dconf.dim_h,
            width=self.dconf.dim_w,
            lstm_channels=self.mconf.lstm_channels,
            all_hidden=False,
            mode='merge',
            dropout_rate=self.mconf.lstm_dropout,
            last_conv=True,
            conv_channels=2,
        )
        self.conv_lstm_t = ConvLSTM(
            in_channels=self.mconf.lstm_channels,
            height=self.dconf.dim_h,
            width=self.dconf.dim_w,
            lstm_channels=self.mconf.lstm_channels,
            all_hidden=False,
            mode='merge',
            dropout_rate=self.mconf.lstm_dropout,
            last_conv=True,
            conv_channels=2,
        )

        self.time_aware_extnn = ExtNN(
            in_features=self.dconf.ext_dim,
            out_height=1,
            out_width=1,
            out_channels=1,
            inter_features=32,
            map=False,
            relu=False,
            mode='last'
        )

    def forward(self, x, x_ext, y_ext):
        features = self.resnn(x)
        ext = self.extnn(x_ext)

        features = features + ext

        # calc attention using Conv-LSTM
        
        hidden_list = self.conv_lstm(features)
        hidden_list_c = hidden_list[:, :self.mconf.lstm_channels * self.len_local]
        hidden_list_t = hidden_list[:, self.mconf.lstm_channels * self.len_local:]
        features_c = features[:, :self.mconf.lstm_channels * self.len_local]
        features_t = features[:, self.mconf.lstm_channels * self.len_local:]

        attention_c = self.concat_conv_c(features_c, hidden_list_c)
        attention_t = self.concat_conv_t(features_t, hidden_list_t)

        self.spatial_attention = torch.cat([attention_c, attention_t], 1)

        phase_c = features_c * (1 + attention_c)
        phase_t = features_t * (1 + attention_t)

        pred_c = self.conv_lstm_c(phase_c)
        pred_t = self.conv_lstm_t(phase_t)

        time_aware = self.time_aware_extnn(y_ext)

        self.time_aware_c = torch.sigmoid(time_aware)
        self.time_aware_t = torch.sigmoid(-1 * time_aware)

        time_aware_c = self.time_aware_c.view(-1, 1, 1, 1)
        time_aware_t = self.time_aware_t.view(-1, 1, 1, 1)

        pred = time_aware_c * pred_c + time_aware_t * pred_t
        h = torch.tanh(pred)

        return h
