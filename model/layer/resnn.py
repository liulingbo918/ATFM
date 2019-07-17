import torch
import torch.nn as nn

class ResUnit(nn.Module):
    def __init__(self, filters, bnmode=True):
        super().__init__()
        self.filters = filters
        self.bnmode = bnmode

        self.layer1 = self._bn_relu_conv()
        self.layer2 = self._bn_relu_conv()
    
    def _conv_layer(self):
        return nn.Conv2d(self.filters, self.filters, 3, 1, 1)
    
    def _bn_relu_conv(self):
        layers = []
        if self.bnmode:
            layers.append(nn.BatchNorm2d(self.filters))
        # layers.append(nn.ReLU(inplace=True))
        layers.append(nn.SELU(inplace=True))
        layers.append(self._conv_layer())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        residual = self.layer1(x)
        residual = self.layer2(residual)
        out = residual + x
        return out

class ResNN(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, repetation=1, bnmode=True, splitmode='split', cpt=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.repetation = repetation
        self.bnmode = bnmode

        self.inlist = []
        self.resblocks = nn.ModuleList()
        if splitmode == 'split':
            seq_num = sum(cpt)
            inscale = int(in_channels) // seq_num
            outscale = int(out_channels) // seq_num
            resblock = self.residual_block(inscale, outscale)
            for i in range(seq_num):
                self.inlist.append(inscale)
                self.resblocks.append(resblock)
        elif splitmode == 'split-chans':
            seq_num = sum(cpt) * 2
            inscale = int(in_channels) // seq_num
            outscale = int(out_channels) // seq_num
            resblock = self.residual_block(inscale, outscale)
            for i in range(seq_num):
                self.inlist.append(inscale)
                self.resblocks.append(resblock)
        elif splitmode == 'concat':
            self.inlist.append(in_channels)
            self.resblocks.append(self.residual_block(in_channels, out_channels))
        elif splitmode == 'cpt':
            seq_num = sum(cpt)
            inscale = int(in_channels) // seq_num
            outscale = int(out_channels) // seq_num
            for i in cpt:
                if i > 0:
                    self.inlist.append(i * inscale)
                    self.resblocks.append(self.residual_block(i * inscale, i * outscale))
        elif splitmode == 'cpt-sameoutput':
            seq_num = sum(cpt)
            inscale = int(in_channels) // seq_num
            for i in cpt:
                if i > 0:
                    self.inlist.append(i * inscale)
                    self.resblocks.append(self.residual_block(i * inscale, 2))
        else:
            raise ValueError('Invalid ResNN split mode')    


    def residual_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, self.inter_channels, 3, 1, 1))
        for _ in range(self.repetation):
            layers.append(ResUnit(self.inter_channels, self.bnmode))
        # layers.append(nn.ReLU(inplace=True))
        layers.append(nn.SELU(inplace=True))
        layers.append(nn.Conv2d(self.inter_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        inputs = torch.split(x, split_size_or_sections=self.inlist, dim=1)
        if len(inputs) != len(self.resblocks):
            raise ValueError('Input length and network in_channels are inconsistent')  

        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.resblocks[i](inputs[i]))
        
        return torch.cat(outputs, dim=1)
