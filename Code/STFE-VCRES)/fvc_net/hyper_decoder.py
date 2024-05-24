from fvc_net.basic import *
from fvc_net.basic import *

# from fvc_net.layers.layers import QReLU
from fvc_net.layers.layers import QReLU

class HyperDecoder(nn.Sequential):
    def __init__(
            self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
    ):
        super().__init__()
        self.deconv1=deconv(in_planes, mid_planes, kernel_size=5, stride=2)
        self.relu1=nn.ReLU(inplace=True)
        self.deconv2=deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3=deconv(mid_planes, out_planes, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        return x


class HyperDecoderWithQReLU(nn.Module):
    def __init__(
            self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
    ):
        super().__init__()

        def qrelu(input, bit_depth=8, beta=100):
            return QReLU.apply(input, bit_depth, beta)

        self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
        self.qrelu1 = qrelu
        self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
        self.qrelu2 = qrelu
        self.deconv3 = deconv(mid_planes, out_planes, kernel_size=3, stride=1)
        self.qrelu3 = qrelu

    def forward(self, x):
        x = self.qrelu1(self.deconv1(x))
        x = self.qrelu2(self.deconv2(x))
        x = self.qrelu3(self.deconv3(x))

        return x
