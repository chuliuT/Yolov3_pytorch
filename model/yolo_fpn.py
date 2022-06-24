import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Convolutional


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.upsample(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class FPN_YOLOV3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """

    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLOV3, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        # large
        self.large_conv=nn.Sequential(
            Convolutional(filters_in=fi_0,filters_out=512,kernel_size=1,stride=1,pad=0,norm='bn'),
            Convolutional(filters_in=512,filters_out=1024,kernel_size=3,stride=1,pad=1,norm='bn'),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn'),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm='bn'),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm='bn'),
        )

        self.conv_l0=Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm='bn')
        self.conv_l1=Convolutional(filters_in=1024, filters_out=fo_0, kernel_size=1, stride=1, pad=0, act=False)


        self.conv_l=Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn')
        self.upsample0=Upsample(scale_factor=2)
        self.route0=Route()

        # medium
        self.medium_conv=nn.Sequential(
            Convolutional(filters_in=fi_1+256,filters_out=256,kernel_size=1,stride=1,pad=0,norm='bn'),
            Convolutional(filters_in=256,filters_out=512,kernel_size=3,stride=1,pad=1,norm='bn'),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn'),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm='bn'),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm='bn'),
        )

        self.conv_m0=Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm='bn')
        self.conv_m1=Convolutional(filters_in=512, filters_out=fo_1, kernel_size=1, stride=1, pad=0, act=False)


        self.conv_m=Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn')
        self.upsample1=Upsample(scale_factor=2)
        self.route1=Route()
        # small
        self.small_conv=nn.Sequential(
            Convolutional(filters_in=fi_2+128,filters_out=128,kernel_size=1,stride=1,pad=0,norm='bn'),
            Convolutional(filters_in=128,filters_out=256,kernel_size=3,stride=1,pad=1,norm='bn'),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn'),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm='bn'),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm='bn'),
        )

        self.conv_s0=Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm='bn')
        self.conv_s1=Convolutional(filters_in=256, filters_out=fo_2, kernel_size=1, stride=1, pad=0, act=False)


    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        l0=self.large_conv(x0)
        out0=self.conv_l0(l0)
        out0=self.conv_l1(out0)

        m0=self.conv_l(l0)
        m0=self.upsample0(m0)
        x1=self.route0(x1,m0)
        m0=self.medium_conv(x1)
        out1=self.conv_m0(m0)
        out1=self.conv_m1(out1)

        s0=self.conv_m(m0)
        s0=self.upsample1(s0)
        x2=self.route1(x2,s0)
        s0=self.small_conv(x2)
        out2=self.conv_s0(s0)
        out2=self.conv_s1(out2)

        return out2, out1, out0  # small, medium, large
