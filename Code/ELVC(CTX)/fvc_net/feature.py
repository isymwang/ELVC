from fvc_net.basic import *
from fvc_net.layer import *
in_channels = 3
out_channel = 64


class feature_exactnet(nn.Module):
    def __init__(self):
        super(feature_exactnet, self).__init__()
        self.conv1=conv(3, out_channel_F, kernel_size=5, stride=2)
        self.resb1=ResBlock(out_channel_F, out_channel_F)
        self.resb2=ResBlock(out_channel_F, out_channel_F)
        self.resb3=ResBlock(out_channel_F, out_channel_F)

    def forward(self,x):
        x1=self.conv1(x)
        x2=self.resb1(x1)
        x3=self.resb2(x2)
        x4=self.resb3(x3)
        return x4+x1

class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channel, 3, stride=2,padding=1)
        self.res_block1 = ResBlock(out_channel,out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=2,padding=1)
        self.res_block2 = ResBlock(out_channel,out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, stride=2,padding=1)
        self.res_block3 = ResBlock(out_channel,out_channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)
        layer1 = self.res_block1(layer1)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)
        layer2 = self.res_block2(layer2)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)
        layer3 = self.res_block3(layer3)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3

class Feature_reconsnet(nn.Module):
    def __init__(self):
        super(Feature_reconsnet, self).__init__()
        self.resb1 = ResBlock(out_channel_F, out_channel_F)
        self.resb2 = ResBlock(out_channel_F, out_channel_F)
        self.resb3 = ResBlock(out_channel_F, out_channel_F)
        self.deconv1= deconv(out_channel_F, 3, kernel_size=5, stride=2)
    def forward(self,x):
        x1=self.resb1(x)
        x2=self.resb2(x1)
        x3=self.resb3(x2)
        x4=self.deconv1(x3+x)
        return x4



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = torch.mean(x, dim=(-1, -2))
        y = self.fc(y)
        return x * y[:, :, None, None]

class ConvBlockResidual(nn.Module):
    def __init__(self, ch_in, ch_out, se_layer=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            SELayer(ch_out) if se_layer else nn.Identity(),
        )
        self.up_dim = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up_dim(x)
        return x2 + x1


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlockResidual(ch_in=in_ch, ch_out=32)
        self.conv2 = ConvBlockResidual(ch_in=32, ch_out=64)
        self.conv3 = ConvBlockResidual(ch_in=64, ch_out=128)

        self.context_refine = nn.Sequential(
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
            ResBlock(128, 0),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = ConvBlockResidual(ch_in=128, ch_out=64)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = ConvBlockResidual(ch_in=64, ch_out=out_ch)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2