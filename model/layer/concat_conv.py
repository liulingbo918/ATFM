import torch
import torch.nn as nn

class ConcatConv(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, inter_channels, relu_conv=False, seq_len=None):
        super().__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.in_channels = in_channels1 + in_channels2
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.relu_conv = relu_conv
        self.seq_len = seq_len
        if seq_len is not None:
            self.in_channels //= seq_len
            self.out_channels //= (seq_len)
        
            self.model = nn.ModuleList()
            for _ in range(self.seq_len):
                self.model.append(self._layer())
        else:
            self.model = self._layer()

    
    def _conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def _layer(self):
        if not self.relu_conv:
            return self._conv_layer(self.in_channels, self.out_channels)
        else:
            conv1 = self._conv_layer(self.in_channels, self.inter_channels)
            # relu = nn.ReLU(inplace=True)
            selu = nn.SELU(inplace=True)
            conv2 = self._conv_layer(self.inter_channels, self.out_channels)
            # return nn.Sequential(conv1, relu, conv2)
            return nn.Sequential(conv1, selu, conv2)

    def forward(self, x, y):
        if self.seq_len is not None:
            x_splited_list = torch.split(x, self.in_channels1//self.seq_len, dim=1) 
            y_splited_list = torch.split(y, self.in_channels2//self.seq_len, dim=1)

            outlist = []
            for i in range(self.seq_len):
                input = torch.cat([x_splited_list[i], y_splited_list[i]], dim=1)
                outlist.append(self.model[i](input))
            return torch.cat(outlist, dim=1)
        else:
            input = torch.cat([x, y], dim=1)
            return self.model(input)
