import torch
import torch.nn as nn
import numpy as np
from model.model_utils import Convolutional, ResidualBlock


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

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
        # self.res_block_5_1 = ResidualBlock(filters_in=1024, filters_out=1024, filters_medium=512)
        # self.res_block_5_2 = ResidualBlock(filters_in=1024, filters_out=1024, filters_medium=512)
        # self.res_block_5_3 = ResidualBlock(filters_in=1024, filters_out=1024, filters_medium=512)

    def forward(self, x):
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
        # x4_2 = self.res_block_5_1(x4_1)
        # x4_3 = self.res_block_5_2(x4_2)
        # x4_4 = self.res_block_5_3(x4_3)  # large

        return x2_8, x3_8, x4_0

    def load_darknet_weights(self, weight_file, cutoff=74):
        print("load darknet weights : ", weight_file)

        with open(weight_file, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        try:
            for count, m in enumerate(self.modules()):
                # print(count)

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
    model = Darknet53()
    x = torch.rand(2, 3, 416, 416)
    model.load_darknet_weights('E:\Yolov3_pytorch_new\Yolov3_pytorch\darknet53.conv.74')
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
