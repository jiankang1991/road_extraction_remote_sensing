
# revise it according to
# https://github.com/snakers4/spacenet-three/blob/master/src/LinkNet.py
# TO DO
"""
all credits to ternaus and albu
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

nonlinearity = nn.ReLU

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class AlbuNet50(nn.Module):

    def __init__(self, num_classes=1, num_channels=3, pretrained=False):
        super().__init__()

        self.num_classes = num_classes
        filters = [64, 256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.pool = nn.MaxPool2d(2, 2)
        if num_channels==3:
            self.encoder1 = nn.Sequential(resnet.conv1,
                                          resnet.bn1,
                                          resnet.relu,
                                          resnet.maxpool)
        else:
            self.encoder1 = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                                          resnet.bn1,
                                          resnet.relu,
                                          resnet.maxpool)

        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # self.center = DecoderBlock(filters[4], filters[3])
        self.center = DecoderBlockV2(filters[4], 256, 128)

        # Decoder
        self.decoder5 = DecoderBlockV2(filters[4]+128, 256, 128)
        self.decoder4 = DecoderBlockV2(filters[3]+128, 128, 64)
        self.decoder3 = DecoderBlockV2(filters[2]+64, 128, 64)
        self.decoder2 = DecoderBlockV2(filters[1]+64, 64, filters[0])
        self.decoder1 = DecoderBlockV2(filters[0], 64, 32)

        # self.decoder5 = DecoderBlock(filters[4]+filters[3], filters[2])
        # self.decoder4 = DecoderBlock(filters[3]+filters[2], filters[2])
        # self.decoder3 = DecoderBlock(filters[2]+filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1]+filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], 32)

        # Final Classifier
        self.decoder0 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu = nonlinearity(inplace=True)
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x) # 3x256x256 ==> 64x64x64
        e2 = self.encoder2(e1) # 64x64x64 ==> 256x64x64
        e3 = self.encoder3(e2) # 256x64x64 ==> 512x32x32
        e4 = self.encoder4(e3) # 512x32x32 ==> 1024x16x16
        e5 = self.encoder5(e4) # 1024x16x16 ==> 2048x8x8

        center = self.center(self.pool(e5))

        de5 = self.decoder5(torch.cat([center, e5], 1))
        de4 = self.decoder4(torch.cat([de5, e4], 1))
        de3 = self.decoder3(torch.cat([de4, e3], 1))
        de2 = self.decoder2(torch.cat([de3, e2], 1))
        de1 = self.decoder1(de2)
        
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(de1), dim=1)
        else:
            x_out = self.final(de1)

        return x_out



if __name__ == '__main__':

    input1 = torch.randn(16,3,256,256)
    input1 = Variable(input1)

    model = AlbuNet50(pretrained=True)

    output1 = model(input1)

    print(output1.size())



















