import torch
import torch.nn as nn

class ExtNN(nn.Module):
    def __init__(self, in_features, out_height, out_width, out_channels, 
                 inter_features=10, map=True, relu=True, mode='inter', 
                 dropout_rate=0):
        super().__init__()
        self.in_features = in_features
        self.out_height = out_height
        self.out_width = out_width
        self.out_channels = out_channels
        self.inter_features = inter_features
        self.map = map
        self.relu = relu
        self.mode = mode
        self.dropout_rate = dropout_rate

        self.out_features = self.out_height * self.out_width * self.out_channels

        self.model = self.external_block()
    
    def external_block(self):
        layers = []
        layers.append(nn.Linear(self.in_features, self.inter_features))
        # layers.append(nn.ReLU(inplace=True))
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.SELU(inplace=True))
        layers.append(nn.Linear(self.inter_features, self.out_features))
        if self.relu:
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.SELU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.mode == 'inter':
            inputs = torch.split(x, 1, dim=1)
            exts = []
            for input in inputs:
                input = input.squeeze(1)
                out = self.model(input)
                if self.map:
                    out = out.view(-1, self.out_channels, self.out_height, self.out_width)
                exts.append(out)
            return torch.cat(exts, 1)
        else:
            out = self.model(x)
            if self.map:
                out = out.view(-1, self.out_channels, self.out_height, self.out_width)
            return out
