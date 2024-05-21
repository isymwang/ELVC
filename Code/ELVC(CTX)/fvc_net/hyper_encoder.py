from fvc_net.basic import *


class HyperEncoder(nn.Sequential):
    def __init__(
            self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
    ):
        super().__init__()
        self.covn1=conv(in_planes, mid_planes, kernel_size=3, stride=1)
        self.relu1=nn.ReLU(inplace=True)
        self.covn2=conv(mid_planes, mid_planes, kernel_size=5, stride=2)
        self.relu2=nn.ReLU(inplace=True)
        self.covn3=conv(mid_planes, mid_planes, kernel_size=5, stride=2)

    def forward(self, x):
        x=self.relu1(self.covn1(x))
        x=self.relu2(self.covn2(x))
        x=self.covn3(x)
        return x