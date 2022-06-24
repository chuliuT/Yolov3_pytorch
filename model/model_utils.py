import torch.nn as nn

class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, norm=None,act=True):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.use_act=act
        self.conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                              stride=stride, padding=pad, bias=not norm)

        if norm:
            self.bn = nn.BatchNorm2d(num_features=filters_out)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, filters_in, filters_out, filters_medium):
        super(ResidualBlock, self).__init__()
        self.conv1 = Convolutional(filters_in=filters_in, filters_out=filters_medium, kernel_size=1, stride=1, pad=0,
                                   norm="bn")
        self.conv2 = Convolutional(filters_in=filters_medium, filters_out=filters_out, kernel_size=3, stride=1, pad=1,
                                   norm="bn")

    def forward(self, x):
        r = self.conv1(x)
        r = self.conv2(r)
        out = x + r
        return out
