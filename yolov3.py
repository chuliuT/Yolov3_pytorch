import sys

sys.path.append("..")

# AbsolutePath = os.path.abspath(__file__)           #将相对路径转换成绝对路径
# SuperiorCatalogue = os.path.dirname(AbsolutePath)   #相对路径的上级路径
# BaseDir = os.path.dirname(SuperiorCatalogue)        #在“SuperiorCatalogue”的基础上在脱掉一层路径，得到我们想要的路径。
# sys.path.insert(0,BaseDir)                          #将我们取出来的路径加入

import torch.nn as nn
import torch
from model.backbone import Darknet53
from model.backbone_mobile import MobileNet
from model.yolo_fpn import FPN_YOLOV3
from model.yolo_head import Yolo_head
from model.model_utils import Convolutional,ResidualBlock
import config.yolov3_config_voc as cfg
import numpy as np


# from utils.tools import *


class Yolov3(nn.Module):

    def __init__(self, init_weights=True):
        super(Yolov3, self).__init__()
        self.anchors = torch.FloatTensor(cfg.MODEL['ANCHORS'])
        self.strides = torch.FloatTensor(cfg.MODEL['STRIDES'])
        self.nC = cfg.DATA['NUM']
        self.out_channel = cfg.MODEL['ANCHORS_PER_SCLAE'] * (self.nC + 5)

        self.conv = Convolutional(filters_in=3, filters_out=32, kernel_size=3, stride=1, pad=1, norm='bn')

        self.conv_1_0 = Convolutional(filters_in=32, filters_out=64, kernel_size=3, stride=2, pad=1, norm='bn')
        #  1
        self.res_block_1_0 = ResidualBlock(filters_in=64, filters_out=64, filters_medium=32)

        self.conv_2_0 = Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=2, pad=1, norm='bn')
        #  2
        self.res_block_2_0 = ResidualBlock(filters_in=128, filters_out=128, filters_medium=64)
        self.res_block_2_1 = ResidualBlock(filters_in=128, filters_out=128, filters_medium=64)

        self.conv_3_0 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=2, pad=1, norm='bn')
        #  8
        self.res_block_3_0 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)
        self.res_block_3_1 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)
        self.res_block_3_2 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)
        self.res_block_3_3 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)
        self.res_block_3_4 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)
        self.res_block_3_5 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)
        self.res_block_3_6 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)
        self.res_block_3_7 = ResidualBlock(filters_in=256, filters_out=256, filters_medium=128)

        self.conv_4_0 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=2, pad=1, norm='bn')
        #  8
        self.res_block_4_0 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)
        self.res_block_4_1 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)
        self.res_block_4_2 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)
        self.res_block_4_3 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)
        self.res_block_4_4 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)
        self.res_block_4_5 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)
        self.res_block_4_6 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)
        self.res_block_4_7 = ResidualBlock(filters_in=512, filters_out=512, filters_medium=256)

        self.conv_5_0 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn')
        #  4
        self.res_block_5_0 = ResidualBlock(filters_in=1024, filters_out=1024, filters_medium=512)
        self.res_block_5_1 = ResidualBlock(filters_in=1024, filters_out=1024, filters_medium=512)
        self.res_block_5_2 = ResidualBlock(filters_in=1024, filters_out=1024, filters_medium=512)
        self.res_block_5_3 = ResidualBlock(filters_in=1024, filters_out=1024, filters_medium=512)

        # self.backbone = Darknet53()
        # self.backbone = MobileNet()
        self.fpn = FPN_YOLOV3([1024, 512, 256], [self.out_channel, self.out_channel, self.out_channel])

        self.head_s = Yolo_head(self.nC, self.anchors[0], self.strides[0])
        self.head_m = Yolo_head(self.nC, self.anchors[1], self.strides[1])
        self.head_l = Yolo_head(self.nC, self.anchors[2], self.strides[2])

        if init_weights:
            self.__init_weights()

    def forward(self, x):
        out = []

        # xs, xm, xl = self.backbone(x)

        x = self.conv(x)

        x0_0 = self.conv_1_0(x)
        x0_1 = self.res_block_1_0(x0_0)

        x1_0 = self.conv_2_0(x0_1)
        x1_1 = self.res_block_2_0(x1_0)
        x1_2 = self.res_block_2_1(x1_1)

        x2_0 = self.conv_3_0(x1_2)
        x2_1 = self.res_block_3_0(x2_0)
        x2_2 = self.res_block_3_1(x2_1)
        x2_3 = self.res_block_3_2(x2_2)
        x2_4 = self.res_block_3_3(x2_3)
        x2_5 = self.res_block_3_4(x2_4)
        x2_6 = self.res_block_3_5(x2_5)
        x2_7 = self.res_block_3_6(x2_6)
        x2_8 = self.res_block_3_7(x2_7)  # small

        x3_0 = self.conv_4_0(x2_8)
        x3_1 = self.res_block_4_0(x3_0)
        x3_2 = self.res_block_4_1(x3_1)
        x3_3 = self.res_block_4_2(x3_2)
        x3_4 = self.res_block_4_3(x3_3)
        x3_5 = self.res_block_4_4(x3_4)
        x3_6 = self.res_block_4_5(x3_5)
        x3_7 = self.res_block_4_6(x3_6)
        x3_8 = self.res_block_4_7(x3_7)  # medium

        x4_0 = self.conv_5_0(x3_8)
        x4_1 = self.res_block_5_0(x4_0)
        x4_2 = self.res_block_5_1(x4_1)
        x4_3 = self.res_block_5_2(x4_2)
        x4_4 = self.res_block_5_3(x4_3)  # large

        xs, xm, xl=x2_8,x3_8,x4_4

        xs, xm, xl = self.fpn(xl, xm, xs)
        # print(xs.shape)
        # print(xm.shape)
        # print(xl.shape)
        out.append(self.head_s(xs))
        out.append(self.head_m(xm))
        out.append(self.head_l(xl))
        # print(out[0][0].shape)
        # print(out[1][0].shape)
        # print(out[2][0].shape)

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        # cnt=0
        for m in self.modules():
            # cnt+=1
            # print(cnt)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))

    def load_darknet_weights(self, weight_file, cutoff=74):
        print("load darknet weights : ", weight_file)

        with open(weight_file, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        try:
            for count, m in enumerate(self.modules()):
                # if cutoff==count:
                #     break
                if isinstance(m, nn.Conv2d):
                    conv_layer = m
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
                    print("loading weight {}".format(conv_layer))
                    if m.bias is not None:
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                    print("initing {}".format(m))

                elif isinstance(m, nn.BatchNorm2d):
                    bn_layer = m
                    num_b = bn_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    print("load weight {}".format(bn_layer))

                elif isinstance(m, ResidualBlock):

                    conv_layer = m.conv1.conv
                    if m.conv1.norm == 'bn':
                        bn_layer = m.conv1.bn
                        num_b = bn_layer.bias.numel()
                        bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                        bn_layer.bias.data.copy_(bn_b)
                        ptr += num_b
                        bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                        bn_layer.weight.data.copy_(bn_w)
                        ptr += num_b
                        bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_rm)
                        ptr += num_b
                        # Running Var
                        bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_rv)
                        ptr += num_b
                        print("load weight {}".format(bn_layer))
                    else:
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                        # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    print(count, conv_layer)
                    conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w

                    conv_layer = m.conv2.conv
                    if m.conv2.norm == 'bn':
                        bn_layer = m.conv2.bn
                        num_b = bn_layer.bias.numel()
                        bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                        bn_layer.bias.data.copy_(bn_b)
                        ptr += num_b
                        bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                        bn_layer.weight.data.copy_(bn_w)
                        ptr += num_b
                        bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_rm)
                        ptr += num_b
                        # Running Var
                        bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_rv)
                        ptr += num_b
                        print("load weight {}".format(bn_layer))
                    else:
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                        # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    print(count, conv_layer)
                    conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w

                    print("loading weight {}".format(conv_layer))

                elif isinstance(m, Convolutional):

                    conv_layer = m.conv
                    if m.norm == 'bn':
                        bn_layer = m.bn
                        num_b = bn_layer.bias.numel()
                        bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                        bn_layer.bias.data.copy_(bn_b)
                        ptr += num_b
                        bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                        bn_layer.weight.data.copy_(bn_w)
                        ptr += num_b
                        bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_rm)
                        ptr += num_b
                        # Running Var
                        bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_rv)
                        ptr += num_b
                        print("load weight {}".format(bn_layer))
                    else:
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                        # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    print(count, conv_layer)
                    conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w

                    print("loading weight {}".format(conv_layer))
                else:
                    print(count, "|", m)
        except Exception as e:
            print(count,m)
            print(e)


if __name__ == '__main__':
    net = Yolov3()
    # print(net)
    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img)
    net.load_darknet_weights('yolov3.weights')
    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
    print("load done")
