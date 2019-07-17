import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq_adaptor import split_cpt

class ConvGate(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16, peephole_conn=True):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width 
        self.lstm_channels = lstm_channels 
        self.peephole_conn = peephole_conn 

        self.conv_x = self._conv_layer(in_channels)
        self.conv_h = self._conv_layer(lstm_channels)

        if peephole_conn:
            self.w = nn.Parameter(torch.Tensor(lstm_channels, height, width))
            self.b = nn.Parameter(torch.Tensor(lstm_channels, 1, 1))
            nn.init.kaiming_normal_(self.w.data, a=0, mode='fan_in')

    def _linear(self, x):
        return x * self.w + self.b

    def _conv_layer(self, in_channels):
        return nn.Conv2d(in_channels, self.lstm_channels, 3, 1, 1, bias=True) # has bias

    def forward(self, input, state):
        hidden_state, cell_state = state
        convx = self.conv_x(input)
        convh = self.conv_h(hidden_state)
        if self.peephole_conn:
            conv = convx + convh + self._linear(cell_state)
            return torch.sigmoid(conv)
        else:
            conv = convx + convh
            return torch.tanh(conv)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width 
        self.lstm_channels = lstm_channels 
    
        self.f_gate = ConvGate(in_channels, height, width, lstm_channels)
        self.i_gate = ConvGate(in_channels, height, width, lstm_channels)
        self.c_gate = ConvGate(in_channels, height, width, lstm_channels, False) 
        # The naming of 'c_gate' may be confusing, but it's just like the gate's operation.
        self.o_gate = ConvGate(in_channels, height, width, lstm_channels)

    def forward(self, input, state):
        hidden_pre, cell_pre = state
        f = self.f_gate(input, state)
        i = self.i_gate(input, state)
        cell_cur = f * cell_pre + i * self.c_gate(input, state)
        o = self.o_gate(input, (hidden_pre, cell_cur))
        hidden_cur = o * torch.tanh(cell_cur)

        return hidden_cur, cell_cur

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width 
        self.lstm_channels = lstm_channels

        self.z_conv = self._conv_layer(in_channels+lstm_channels)
        self.r_conv = self._conv_layer(in_channels+lstm_channels)
        self.h_conv = self._conv_layer(in_channels+lstm_channels)

    def _conv_layer(self, in_channels):
        return nn.Conv2d(in_channels, self.lstm_channels, 3, 1, 1, bias=True) # has bias

    def forward(self, input, state):
        hidden_pre = state
        mix_input = torch.cat([hidden_pre, input], dim=1)
        z_t = torch.sigmoid(self.z_conv(mix_input))
        r_t = torch.sigmoid(self.r_conv(mix_input))
        mix_input = torch.cat([r_t * hidden_pre, input], dim=1)
        h_t_hat = torch.tanh(self.h_conv(mix_input))
        h_t = (1 - z_t) * hidden_pre + z_t * h_t_hat
        return h_t

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, height, width, lstm_channels=16, all_hidden=False,
                 mode='merge', cpt=None, dropout_rate=0.5, last_conv=False, 
                 conv_channels=None, gru=False):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width 
        self.lstm_channels = lstm_channels 
        self.all_hidden = all_hidden
        self.mode = mode
        self.cpt = cpt
        self.dropout_rate = dropout_rate
        self.last_conv = last_conv
        self.conv_channels = conv_channels
        self.gru = gru

        if gru:
            self._lstm_cell = ConvGRUCell(in_channels, height, width, lstm_channels)
        else:
            self._lstm_cell = ConvLSTMCell(in_channels, height, width, lstm_channels)
        if last_conv:
            if self.conv_channels is None:
                 raise ValueError('Parameter Out Channel is needed to enable last_conv')
   
            self._conv_layer = nn.Conv2d(lstm_channels, conv_channels, 3, 1, 1, bias=True)

        if dropout_rate > 0:
            self._dropout_layer = nn.Dropout2d(dropout_rate)

    def lstm_layer(self, inputs):
        n_in, c_in, h_in, w_in = inputs.size()
        if self.gru:
            state = torch.zeros(n_in, self.lstm_channels, h_in, w_in).cuda()
        else:
            state = (torch.zeros(n_in, self.lstm_channels, h_in, w_in).cuda(),
                     torch.zeros(n_in, self.lstm_channels, h_in, w_in).cuda())
        seq = torch.split(inputs, self.in_channels, dim=1)
        hiddent_list = []
        for idx, input in enumerate(seq[::-1]): # using reverse order
            state = self._lstm_cell(input, state)
            if self.gru:
                hidden = state
            else:
                hidden = state[0]

            if self.last_conv:
                if self.conv_channels is None:
                    raise ValueError('Parameter Out Channel is needed to enable last_conv')
                hidden = self._conv_layer(hidden)

            hiddent_list.append(hidden)
        
        if not self.all_hidden:
            return hiddent_list[-1]
        else:
            hiddent_list.reverse()
            return torch.cat(hiddent_list, 1)

    def forward(self, inputs):
        if self.dropout_rate > 0:
            inputs = self._dropout_layer(inputs)
        if self.mode == 'merge':
            output = self.lstm_layer(inputs)
            return output
        elif self.mode == 'cpt':
            if self.cpt is None:
                raise ValueError('Parameter \'cpt\' is required in mode \'cpt\' of ConvLSTM')
            cpt_seq = split_cpt(inputs, self.cpt)
            output_list = [
                self.lstm_layer(input_) for input_ in cpt_seq
            ] 
            output = torch.cat(output_list, 1)
            return output
        else:
            raise('Invalid LSTM mode: '+self.mode)
