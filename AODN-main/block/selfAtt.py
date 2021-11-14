import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

class SelfAttentionSpa(nn.Module):
    def __init__(self, in_channels=120, out_channels=120):
        super().__init__()
        self.conv1= nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.softmax =nn.Softmax(dim=1)
    def forward(self, feature):
        feature_spac=self.conv1(feature)
        _,c,h,w=feature.size()
        feature_spas=torch.reshape(feature_spac,(-1,h*w,c))# h*w c

        feature_spat=torch.transpose(feature_spas, 1, 2)#c h*w
        spectral_matmul_1=torch.matmul(feature_spas, feature_spat)# h*w h*w
        spectral_softmax=self.softmax(spectral_matmul_1)

        spectral_matmul_2 = torch.matmul(spectral_softmax,feature_spas)#h*w c

        spectral_reshape=torch.reshape(spectral_matmul_2,(-1,c,h,w))
        print(spectral_reshape.shape)
        return self.conv2(feature+spectral_reshape)

class SelfAttentionSpe(nn.Module):
    def __init__(self, in_channels=120, out_channels=120):
        super().__init__()
        self.conv2= nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.softmax =nn.Softmax(dim=1)
    def forward(self, feature):
        _,c,h,w=feature.size()
        feature_spas=torch.reshape(feature,(-1,h*w,c))# h*w c

        feature_spat=torch.transpose(feature_spas, 1, 2)#c h*w
        spectral_matmul_1=torch.matmul(feature_spas, feature_spat)# h*w h*w
        spectral_softmax=self.softmax(spectral_matmul_1)

        spectral_matmul_2 = torch.matmul(spectral_softmax,feature_spas)#h*w c

        spectral_reshape=torch.reshape(spectral_matmul_2,(-1,c,h,w))
        print(spectral_reshape.shape)
        return self.conv2(feature+spectral_reshape)

class SelfAttention(nn.Module):
    def __init__(self, in_channels=120, out_channels=120):
        super().__init__()
        self.spa=SelfAttentionSpa(in_channels=in_channels, out_channels=in_channels)
        self.spe=SelfAttentionSpe(in_channels=in_channels, out_channels=in_channels)
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
    def forward(self, feature):
        feature_spa = self.spa(feature)
        feature_spe = self.spe(feature)

        return self.conv(torch.cat((feature_spa, feature_spe), dim=1))
if __name__ =='__main__':
    # selfAttSpa=SelfAttentionSpa()
    # summary(selfAttSpa.cuda(),input_size=[(120,20,20)])
    # selfAttSpe=SelfAttentionSpe()
    # summary(selfAttSpe.cuda(),input_size=[(120,20,20)])
    selfAtt=SelfAttention(120,80)
    summary(selfAtt.cuda(),input_size=[(120,20,20)])
