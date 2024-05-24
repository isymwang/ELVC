from fvc_net.basic import *
from fvc_net.layers.gdn import *

class EncoderWithResblock(nn.Sequential):
    def __init__(self, in_planes: int=128, out_planes: int=128):
        super().__init__()
        self.conv1=conv(in_planes, out_planes, kernel_size=3, stride=2)
        self.conv2=conv(out_planes, out_planes, kernel_size=3, stride=2)
        self.conv3 = conv(out_planes, out_planes, kernel_size=3, stride=2)
        self.resb=ResBlock(out_planes, out_planes)


    def forward(self, x):
        x1=self.conv1(x)

        x2=self.resb(x1)
        x3=self.resb(x2)
        x4=self.resb(x3)

        x5=self.conv2(x1+x4)
        x6=self.resb(x5)
        x7=self.resb(x6)
        x8=self.resb(x7)

        x9=self.conv3(x5+x8)
        x10=self.resb(x9)
        x11=self.resb(x10)
        x12=self.resb(x11)

        x13=self.conv3(x9+x12)
        return x13




class Analysis_net(nn.Module):
    '''
    Compress residual
    '''
    def __init__(self, in_planes: int=128, out_planes: int=128):
        super(Analysis_net, self).__init__()

        self.resb = ResBlock(out_planes, out_planes)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)



        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)

        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)


        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)


    def forward(self, x):
        x = self.leakyrelu(self.resb(self.gdn1(self.conv1(x))))
        x = self.leakyrelu(self.resb(self.gdn2(self.conv2(x))))
        x = self.leakyrelu(self.resb(self.gdn3(self.conv3(x))))
        return self.conv4(x)
