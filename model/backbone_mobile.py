import torch
import torch.nn as nn
def conv_bn(in_ch,out_ch,s):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=s,padding=1,bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True))

def conv_dw(in_ch,out_ch,s):
    return nn.Sequential(nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=s,padding=1,groups=in_ch,bias=False),
                        nn.BatchNorm2d(in_ch),
                        nn.ReLU(True),
                        nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(True))


class MobileNet(nn.Module):

    def __init__(self, num_class=1000):
        super(MobileNet, self).__init__()

        self.feature = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),

        )

    def forward(self, x):
        out=[]
        out_index=[5,8,13]
        for index,layer in enumerate(self.feature):
            x=layer(x)
            if index in out_index:
                out.append(x)
            # print(index,x.shape)
        return out


if __name__ == '__main__':
    input_image=torch.rand(2,3,416,416)
    net = MobileNet(1000)

    out = net(input_image)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)