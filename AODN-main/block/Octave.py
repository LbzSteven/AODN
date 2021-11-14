import torch.nn as nn
import torch
import torch.nn.functional as F

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,groups=1
                 , bias=False):
        super(OctaveConv, self).__init__()
        # kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = nn.Conv2d(in_channels=int(alpha * in_channels), out_channels=int(alpha * out_channels),
                                   kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,groups=groups,  bias=bias)
        self.l2h = nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, stride, padding, dilation,groups,  bias)
        self.h2l = nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, stride, padding, dilation,groups,  bias)
        self.h2h = nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, X_h, X_l):
        # X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l


class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,groups=1,
                  bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        # kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, stride, padding, dilation,groups,  bias)
        self.h2h = nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, stride, padding, dilation,groups,  bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,groups=1,
                  bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        # kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, stride, padding, dilation,groups,  bias)
        self.h2h = nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, stride, padding, dilation,groups,  bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, X_h, X_l):
        # X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)

        X_h = X_h2h + X_l2h

        return X_h

class OctaveCBR(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=1, dilation=1,
                  bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCBR, self).__init__()
        self.conv = OctaveConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation,  bias)
        self.bn_h = norm_layer(int(out_channels*(1-alpha)))
        self.bn_l = norm_layer(int(out_channels*alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l):
        x_h, x_l = self.conv(x_h, x_l)
        x_h = self.relu((x_h))
        x_l = self.relu((x_l))
        return x_h, x_l

class FirstOctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=1, dilation=1,
                  bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCBR, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu((x_h))
        x_l = self.relu((x_l))
        return x_h, x_l


class LastOCtaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,
                  bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCBR, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,  bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l):
        x_h = self.conv(x_h, x_l)
        x_h = self.relu((x_h))
        return x_h

class OctaveCL(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=1, dilation=1,groups=1,
                  bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCL, self).__init__()
        self.conv = OctaveConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation,groups,  bias)
        self.LeakyReLU = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x_h, x_l):
        x_h, x_l = self.conv(x_h, x_l)
        x_h = self.LeakyReLU((x_h))
        x_l = self.LeakyReLU((x_l))
        return x_h, x_l

class FirstOctaveCL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3),alpha=0.5, stride=1, padding=1, dilation=1,groups=1
                 , bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCL, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.LeakyReLU = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.LeakyReLU((x_h))
        x_l = self.LeakyReLU((x_l))
        return x_h, x_l


class LastOCtaveCL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), alpha=0.5, stride=1, padding=1, dilation=1,groups=1,
                  bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCL, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,groups,  bias)
        self.bn_h = norm_layer(out_channels)
        self.LeakyReLU = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x_h, x_l):
        x_h = self.conv(x_h, x_l)
        x_h = self.LeakyReLU((x_h))
        return x_h