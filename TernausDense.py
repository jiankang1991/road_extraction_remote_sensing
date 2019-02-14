

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
                ConvRelu(in_channels, out_channels)
                # ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class DecoderBlockV3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV3, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=1, stride=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=1, mode='bilinear'),
                ConvRelu(in_channels, out_channels)
                # ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
class DecoderBlockV4(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV4, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=4, output_padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear'),
                ConvRelu(in_channels, out_channels)
                # ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class TernausDense121(nn.Module):

    def __init__(self, num_classes=1, num_channels=3, pretrained=False, is_deconv=True):
        super().__init__()
        self.num_classes = num_classes
        densenet = models.densenet121(pretrained=pretrained)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if num_channels==3:
            self.encoder1 = nn.Sequential(densenet.features[0],
                                          densenet.features[1],
                                          densenet.features[2],
                                          densenet.features[3]) # 'conv0', 'norm0', 'relu0', 'pool0'
        else:
            self.encoder1 = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                          densenet.features[1],
                                          densenet.features[2],
                                          densenet.features[3])

        self.encoder2 = densenet.features[4] # dense block1

        self.encoder2TL = densenet.features[5]
        self.encoder3 = densenet.features[6] # dense block2
        
        self.encoder3TL = densenet.features[7] 
        self.encoder4 = densenet.features[8] # dense block3

        self.encoder4TL = densenet.features[9]
        self.encoder5 = densenet.features[10] # dense block4
        self.encoder5TL = densenet.features[11]

        self.center = DecoderBlockV2(1024, 512, 256, is_deconv)

        self.dec5 = DecoderBlockV2(256+1024, 512, 256, is_deconv) # center + encoder5TL
        self.dec4 = DecoderBlockV2(256+1024, 256, 128, is_deconv) # dec5 + encoder4
        self.dec3 = DecoderBlockV2(128+512, 256, 128, is_deconv) # dec4 + encoder3
        self.dec2 = DecoderBlockV3(128+256, 128, 64, is_deconv) # dec3 + encoder2 
        self.dec1 = DecoderBlockV4(64+64, 64, 32, is_deconv) # dec2 + encoder1

        # self.center = DecoderBlockV2(1024, 64, 64, is_deconv)

        # self.dec5 = DecoderBlockV2(64+1024, 64, 64, is_deconv) # center + encoder5TL
        # self.dec4 = DecoderBlockV2(64+1024, 64, 64, is_deconv) # dec5 + encoder4
        # self.dec3 = DecoderBlockV2(64+512, 64, 64, is_deconv) # dec4 + encoder3
        # self.dec2 = DecoderBlockV3(64+256, 64, 64, is_deconv) # dec3 + encoder2 
        # self.dec1 = DecoderBlockV4(64+64, 64, 32, is_deconv) # dec2 + encoder1

        self.dec0 = ConvRelu(32, 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        # print(enc1.size())

        enc2 = self.encoder2(enc1)
        # print(enc2.size())

        enc2TL = self.encoder2TL(enc2)
        # print(enc2TL.size())
        enc3 = self.encoder3(enc2TL)
        # print(enc3.size())

        enc3TL = self.encoder3TL(enc3)
        # print(enc3TL.size())
        enc4 = self.encoder4(enc3TL)
        # print(enc4.size())

        enc4TL = self.encoder4TL(enc4)
        # print(enc4TL.size())
        enc5 = self.encoder5(enc4TL)
        # print(enc5.size())
        enc5TL = self.encoder5TL(enc5)
        # print(enc5TL.size())

        center = self.center(self.pool(enc5TL))
        # print(center.size())
        
        dec5 = self.dec5(torch.cat([center, enc5TL], 1))
        # print(dec5.size())
        dec4 = self.dec4(torch.cat([dec5, enc4], 1))
        # print(dec4.size())
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        # print(dec3.size())
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        # print(dec2.size())
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        # print(dec1.size())
        dec0 = self.dec0(dec1)

        logits = self.final(dec0)

        return logits


class TernausDense169(nn.Module):

    def __init__(self, num_classes=1, num_channels=3, pretrained=False, is_deconv=True):
        super().__init__()
        self.num_classes = num_classes
        densenet = models.densenet169(pretrained=pretrained)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if num_channels==3:
            self.encoder1 = nn.Sequential(densenet.features[0],
                                          densenet.features[1],
                                          densenet.features[2],
                                          densenet.features[3]) # 'conv0', 'norm0', 'relu0', 'pool0'
        else:
            self.encoder1 = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                          densenet.features[1],
                                          densenet.features[2],
                                          densenet.features[3])

        self.encoder2 = densenet.features[4] # dense block1

        self.encoder2TL = densenet.features[5]
        self.encoder3 = densenet.features[6] # dense block2
        
        self.encoder3TL = densenet.features[7] 
        self.encoder4 = densenet.features[8] # dense block3

        self.encoder4TL = densenet.features[9]
        self.encoder5 = densenet.features[10] # dense block4
        self.encoder5TL = densenet.features[11]

        # self.center = DecoderBlockV2(1664, 512, 256, is_deconv)

        # self.dec5 = DecoderBlockV2(256+1664, 512, 256, is_deconv) # center + encoder5TL
        # self.dec4 = DecoderBlockV2(256+1280, 256, 128, is_deconv) # dec5 + encoder4
        # self.dec3 = DecoderBlockV2(128+512, 256, 128, is_deconv) # dec4 + encoder3
        # self.dec2 = DecoderBlockV3(128+256, 128, 64, is_deconv) # dec3 + encoder2 
        # self.dec1 = DecoderBlockV4(64+64, 64, 32, is_deconv) # dec2 + encoder1

        self.center = DecoderBlockV2(1664, 64, 64, is_deconv)

        self.dec5 = DecoderBlockV2(256+1664, 64, 64, is_deconv) # center + encoder5TL
        self.dec4 = DecoderBlockV2(256+1280, 64, 64, is_deconv) # dec5 + encoder4
        self.dec3 = DecoderBlockV2(128+512, 64, 64, is_deconv) # dec4 + encoder3
        self.dec2 = DecoderBlockV3(128+256, 64, 32, is_deconv) # dec3 + encoder2 
        self.dec1 = DecoderBlockV4(64+64, 32, 32, is_deconv) # dec2 + encoder1

        self.dec0 = ConvRelu(32, 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        # print('enc1', enc1.size())

        enc2 = self.encoder2(enc1)
        # print('enc2', enc2.size())

        enc2TL = self.encoder2TL(enc2)
        # print('enc2TL', enc2TL.size())
        enc3 = self.encoder3(enc2TL)
        # print('enc3', enc3.size())

        enc3TL = self.encoder3TL(enc3)
        # print('enc3TL', enc3TL.size())
        enc4 = self.encoder4(enc3TL)
        # print('enc4', enc4.size())

        enc4TL = self.encoder4TL(enc4)
        # print('enc4TL', enc4TL.size())
        enc5 = self.encoder5(enc4TL)
        # print('enc5', enc5.size())
        enc5TL = self.encoder5TL(enc5)
        # print('enc5TL', enc5TL.size())

        center = self.center(self.pool(enc5TL))
        # print('center', center.size())
        
        dec5 = self.dec5(torch.cat([center, enc5TL], 1))
        # print(dec5.size())
        dec4 = self.dec4(torch.cat([dec5, enc4], 1))
        # print(dec4.size())
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        # print(dec3.size())
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        # print(dec2.size())
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        # print(dec1.size())
        dec0 = self.dec0(dec1)

        logits = self.final(dec0)

        return logits


if __name__ == '__main__':


    input1 = torch.randn(16,3,256,256)
    input1 = Variable(input1)
    model = TernausDense121(num_classes=1, num_channels=3,is_deconv=False,pretrained=True)
    # model = TernausDense169(num_classes=1, num_channels=3,pretrained=True)

    # print(model)
    output1 = model(input1)

    print(output1.size())
    # print(model)
    
    # print(model.densenet.features[0])
    # print(model.encoder1)
    # print(model.encoder2)
    # print(model.encoder2TS)
    # print(model.encoder3)

    # print(model.encoder5)
    # print(model.encoder5TL)

    # print(model.encoder4TL)
    # print(model.encoder3TL)
    # print(model.encoder2TL)


