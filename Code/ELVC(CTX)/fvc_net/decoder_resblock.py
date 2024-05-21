from fvc_net.basic import *
from fvc_net.layers.gdn import *

class DecoderWithResblock(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.resb=ResBlock(in_planes, in_planes)
        self.deconv1=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv2=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv3=deconv(in_planes, out_planes, kernel_size=3, stride=2)

        self.gdn = GDN(out_planes,inverse=True)

    def forward(self, x):
        x=self.deconv1(x)

        x1=self.resb(x)
        x2=self.resb(x1)
        x3=self.resb(x2)
        x4=self.deconv1(x+x3)



        x5=self.resb(x4)
        x6=self.resb(x5)
        x7=self.resb(x6)
        x8=self.deconv2(x4+x7)


        x9=self.resb(x8)
        x10=self.resb(x9)
        x11=self.resb(x10)
        x12=self.deconv3(x11+x8)

        return x12




class DecoderWithResblock_flow(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.resb=ResBlock(in_planes, in_planes)
        self.deconv1=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv2=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv3=deconv(in_planes, out_planes, kernel_size=3, stride=2)

    def forward(self, x):
        x1=self.resb(x)
        x2=self.resb(x1)
        x3=self.resb(x2)
        x4=self.deconv1(x+x3)

        x5=self.resb(x4)
        x6=self.resb(x5)
        x7=self.resb(x6)
        x8=self.deconv2(x4+x7)

        x9=self.resb(x8)
        x10=self.resb(x9)
        x11=self.resb(x10)
        x12=self.deconv3(x8+x11)
        return x12


class Synthesis_net(nn.Module):
    '''
    Decode residual
    '''

    def __init__(self, in_planes: int=128, out_planes: int=128):
        super(Synthesis_net, self).__init__()

        self.resb = ResBlock(out_planes, out_planes)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)


        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (
            math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)


        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)


        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)

        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data,
                                     (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.resb(self.leakyrelu(self.igdn1(self.deconv1(x))))
        x = self.resb(self.leakyrelu(self.igdn2(self.deconv2(x))))
        x = self.resb(self.leakyrelu(self.igdn3(self.deconv3(x))))
        x = self.deconv4(x)
        return x