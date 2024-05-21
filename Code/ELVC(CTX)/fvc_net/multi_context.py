from fvc_net.basic import *
from fvc_net.layers.gdn import *

# class TemporalPriorEncoder(nn.Module):
#     def __init__(self, in_planes: int, out_planes: int):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(out_planes,out_planes,5,stride=2,padding=2)
#         self.gdn1 = GDN(out_planes)
#
#         self.conv2 = nn.Conv2d(out_planes,out_planes,5,stride=2,padding=2)
#         self.gdn2 = GDN(out_planes)
#         self.conv3 =nn.Conv2d(out_planes,out_planes,5,stride=2,padding=2)
#         self.gdn3 = GDN(out_planes)
#         self.conv4=nn.Conv2d(out_planes,out_planes,5,padding=2)
#         self.conv5=nn.Conv2d(out_planes,out_planes*2,5,padding=2)
#
#
#
#     def forward(self, context):
#         feature = self.conv1(context)
#         feature = self.gdn1(feature)
#
#
#         feature = self.conv2(feature)
#         feature = self.gdn2(feature)
#
#
#         feature = self.conv3(feature)
#         feature = self.gdn3(feature)
#
#         feature=self.conv4(feature)
#         feature=self.conv5(feature)
#
#         return feature

class ContextualEncoder(nn.Module):
    def __init__(self, in_planes=128, out_planes=64):
        super().__init__()

        self.conv1 = nn.Conv2d(out_channel_N*2,out_channel_N,kernel_size=5,stride=2,padding=2)
        self.gdn1=GDN(out_channel_N)

        self.resb=ResBlock_LeakyReLU_0_Point_1(out_channel_N)


        self.conv2 = nn.Conv2d(out_channel_N,out_channel_N,kernel_size=5,stride=2,padding=2)
        self.conv3=nn.Conv2d(out_channel_N,out_channel_N*2,kernel_size=5,padding=2)


        self.gdn2= GDN(out_channel_M)

    def forward(self,x):
        feature1 = self.conv1(x)
        feature2 = self.gdn1(feature1)
        feature3=self.resb(feature2)

        feature4=self.conv2(feature3)
        feature5=self.gdn1(feature4)
        feature6=self.resb(feature5)

        feature7=self.conv2(feature6)
        feature8=self.gdn1(feature7)

        feature9=self.conv3(feature8)

        return feature9




