from torchsummary import summary
import torch.nn.init as init
from block.Octave import *
from block.CBAM import ChannelAttention , SpatialAttention
from block.separableConv import SeparableConv3d,SeparableConv2d
from thop import profile
class dn_block(nn.Module):
    def __init__(self,block_num=6,layer_input=80,layer_output=80,alpha=0.2):
        super(dn_block,self).__init__()
        layer_input = layer_input
        layer_output = layer_output
        self.group_list = []
        dense_output = int(layer_output//2)
        self.FirstOct=FirstOctaveCL(layer_input, dense_output, 3, padding=1, alpha=alpha)

        dense_input=dense_output

        for i in range(block_num-2):
            group = OctaveCL(dense_input ,dense_output, 3, padding=1, alpha=alpha)
            self.add_module(name='group_%d' % i, module=group)
            self.group_list.append(group)
            dense_input = dense_input + dense_output
        self.LastOct=LastOCtaveCL(dense_input, layer_output, 3, padding=1, alpha=alpha)
        self.ca1 = ChannelAttention(layer_output)
        self.sa1 = SpatialAttention(7)
    def forward(self,input):
        X_h, X_l = self.FirstOct(input)
        feature_high = [X_h]
        feature_low = [X_l]
        for group in self.group_list:
            inputs_high = torch.cat(feature_high, dim=1)
            inputs_low = torch.cat(feature_low, dim=1)
            X_hi, X_li = group(inputs_high,inputs_low)
            feature_high.append(X_hi)
            feature_low.append(X_li)
        last_inputs_high = torch.cat(feature_high, dim=1)
        last_inputs_low = torch.cat(feature_low, dim=1)
        LastOct = self.LastOct(last_inputs_high,last_inputs_low)
        caout =self.ca1(LastOct) *LastOct
        out =self.sa1(caout) *caout +input
        return out

class AODN(nn.Module):
    def __init__(self,block_num=6,dense_num=6,alpha=0.5,K=24,feature_Channel=20,layer_output=80):
        super(AODN, self).__init__()
        input_Channel=1
        feature_Channel = feature_Channel
        layer_output = layer_output

        self.Spatial_Feature_3 = nn.Sequential(
            nn.Conv2d(input_Channel, feature_Channel, 3, padding=1),

        )
        self.Spatial_Feature_5 = nn.Sequential(
            nn.Conv2d(input_Channel, feature_Channel, 5, padding=2),

        )
        self.Spatial_Feature_7 = nn.Sequential(
            nn.Conv2d(input_Channel, feature_Channel, 7, padding=3),

        )

        self.Spectral_Feature_3 = nn.Sequential(
            nn.Conv3d(1, feature_Channel, (K, 1, 1), 1, (0, 0, 0)),
            nn.Conv3d(feature_Channel, feature_Channel, (1, 3, 3), 1, (0, 1, 1))
        )
        self.Spectral_Feature_5 = nn.Sequential(
            nn.Conv3d(1, feature_Channel, (K, 1, 1), 1, (0, 0, 0)),
            nn.Conv3d(feature_Channel, feature_Channel, (1, 5, 5), 1, (0, 2, 2))
        )
        self.Spectral_Feature_7 = nn.Sequential(
            nn.Conv3d(1, feature_Channel, (K, 1, 1), 1, (0, 0, 0)),
            nn.Conv3d(feature_Channel, feature_Channel, (1, 7, 7), 1, (0, 3, 3))
        )

        self.ca_in = ChannelAttention(feature_Channel * 6)
        self.sa_in = SpatialAttention(7)
        self.concat_in = nn.Sequential(
            nn.Conv2d(feature_Channel*6, layer_output, 3, 1, padding=1),
            nn.LeakyReLU()
        )

        self.dn_block_in = dn_block(dense_num,layer_output,layer_output,alpha)
        self.inner_group_list = []
        for i in range(block_num-1):
            group = dn_block(dense_num ,layer_output, layer_output, alpha)
            self.add_module(name='group_%d' % i, module=group)
            self.inner_group_list.append(group)

        self.ca = ChannelAttention((block_num+1)*layer_output)
        self.sa = SpatialAttention(7)
        self.FR = nn.Conv2d(layer_output * ( block_num + 1), input_Channel, 3, padding=1)
    def forward(self, Spatial, Spectral):
        Spatial_3 = self.Spatial_Feature_3(Spatial)
        Spatial_5 = self.Spatial_Feature_5(Spatial)
        Spatial_7 = self.Spatial_Feature_7(Spatial)

        Spectral_3 = self.Spectral_Feature_3(Spectral)
        Spectral_5 = self.Spectral_Feature_5(Spectral)
        Spectral_7 = self.Spectral_Feature_7(Spectral)

        out1 = F.leaky_relu(torch.cat((Spatial_3, Spatial_5, Spatial_7), dim=1))
        out2 = F.leaky_relu(torch.cat((Spectral_3, Spectral_5, Spectral_7), dim=1)).squeeze(2)

        out3 = torch.cat((out1, out2), dim=1)
        ca_in = self.ca_in(out3) * out3
        sa_in = self.sa_in(ca_in) * ca_in
        out3 =  self.concat_in(sa_in)

        block_in = self.dn_block_in(out3)
        in_feature = block_in
        output_feature = [out3,in_feature]
        for group in self.inner_group_list:
            in_feature = group(in_feature)
            output_feature.append(in_feature)

        concat = torch.cat(output_feature, dim=1)
        caout = concat * self.ca(concat)
        saout = caout * self.sa(caout)
        Residual = self.FR(saout)
        out = Spatial - Residual

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

if __name__ == '__main__':
    K = 64
    dense_num = 4
    alpha=0.2

    input1 = torch.randn(1,1,40, 40)
    input2= torch.randn(1,1,K,40, 40)
    casaoct = AODN(dense_num=dense_num, alpha=alpha, K=K)
    flops, params = profile(casaoct, inputs=(input1,input2,))
    print('flops: {}, params: {}'.format(flops, params))

