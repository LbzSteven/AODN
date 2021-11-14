import torch.nn as nn
from torchsummary import summary

class SeparableConv2d(nn.Module):
    def __init__(self ,in_channels ,out_channels ,kernel_size ,stride=1 ,padding=0 ,dilation=1 ,bias=False):
        super(SeparableConv2d ,self).__init__()

        self.conv1 = nn.Conv2d(in_channels ,in_channels ,kernel_size ,stride ,padding ,dilation ,groups=in_channels
                               ,bias=bias)
        self.pointwise = nn.Conv2d(in_channels ,out_channels ,1 ,1 ,0 ,1 ,1 ,bias=bias)

    def forward(self ,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class commomConv2d(nn.Module):
    def __init__(self ,in_channels ,out_channels ,kernel_size ,stride=1 ,padding=0 ,dilation=1 ,bias=False):
        super(commomConv2d ,self).__init__()

        self.conv1 = nn.Conv2d(in_channels ,out_channels ,kernel_size ,stride ,padding ,dilation ,groups=1
                               ,bias=bias)


    def forward(self ,x):
        x = self.conv1(x)

        return x

class SeparableConv3d(nn.Module):
    def __init__(self ,in_channels ,out_channels ,kernel_size ,stride=1 ,padding=0 ,dilation=1 ,bias=False):
        super(SeparableConv3d ,self).__init__()

        # self.conv1 = nn.Conv3d(in_channels ,in_channels ,kernel_size ,stride ,padding ,dilation ,groups=in_channels
        #                        ,bias=bias)
        # self.pointwise = nn.Conv3d(in_channels ,out_channels ,1 ,1 ,0 ,1 ,1 ,bias=bias)
        self.conv1 = nn.Conv3d(in_channels ,out_channels ,kernel_size=(1,3,3) ,stride=1 ,padding=(0,1,1) ,dilation=1 ,groups=1
                               ,bias=bias)
        self.conv2 = nn.Conv3d(out_channels ,out_channels ,kernel_size=(3,1,1) ,stride=1 ,padding=(1,0,0) ,dilation=1 ,groups=1
                               ,bias=bias)
    def forward(self ,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class commomConv3d(nn.Module):
    def __init__(self ,in_channels ,out_channels ,kernel_size ,stride=1 ,padding=0 ,dilation=1 ,bias=False):
        super(commomConv3d ,self).__init__()

        self.conv1 = nn.Conv3d(in_channels ,out_channels ,kernel_size ,stride ,padding ,dilation ,groups=1
                               ,bias=bias)


    def forward(self ,x):
        x = self.conv1(x)

        return x

if __name__ == '__main__':
    # t=SeparableConv2d(in_channels=20 ,out_channels=10,kernel_size=3,padding=1)
    # summary(t.cuda(), input_size=[(20, 40, 40)])
    # t2=commomConv2d(in_channels=20 ,out_channels=20,kernel_size=3,padding=1)
    # summary(t2.cuda(), input_size=[(20, 40, 40)])
    t3=commomConv3d(in_channels=20 ,out_channels=20,kernel_size=3,padding=1)
    summary(t3.cuda(), input_size=[(20,32, 40, 40)])
    t4=SeparableConv3d(in_channels=20 ,out_channels=20,kernel_size=3,padding=1)
    summary(t4.cuda(), input_size=[(20,32, 40, 40)])