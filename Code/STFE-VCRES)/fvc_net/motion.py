from fvc_net.basic import *
from torch.nn.functional import gumbel_softmax
from fvc_net.layer import *
import torch
from torch import nn
import torch.nn.functional as F

from fvc_net.feature import *


modelspath = 'G:/wangyiming/optical_flow_constrastive_residual_learning/flow_pretrain_np/'

Backward_tensorGrid = [{} for i in range(8)]

def torch_warp(tensorInput, tensorFlow):
    device_id = tensorInput.device.index
    if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda().to(device_id)
            # B, C, H, W = tensorInput.size()
            # xx = torch.arange(0, W).view(1,-1).repeat(H,1)
            # yy = torch.arange(0, H).view(-1,1).repeat(1,W)
            # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
            # yy = yy.view(1,1,H,W).repeat(B,1,1,1)
            # Backward_tensorGrid[device_id][str(tensorFlow.size())] = Variable(torch.cat([xx, yy], 1).float().cuda()).to(device_id)

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# # end

def flow_warp(im, flow):
    warp = torch_warp(im, flow)

    return warp

class Warp_net(nn.Module):
    def __init__(self):
        super(Warp_net, self).__init__()
        channelnum = 64

        self.feature_ext = nn.Conv2d(6, channelnum, 3, padding=1)# feature_ext
        self.f_relu = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.feature_ext.weight.data)
        torch.nn.init.constant_(self.feature_ext.bias.data, 0.0)
        self.conv0 = ResBlock(channelnum, channelnum, 3)#c0
        self.conv0_p = nn.AvgPool2d(2, 2)# c0p
        self.conv1 = ResBlock(channelnum, channelnum, 3)#c1
        self.conv1_p = nn.AvgPool2d(2, 2)# c1p
        self.conv2 = ResBlock(channelnum, channelnum, 3)# c2
        self.conv3 = ResBlock(channelnum, channelnum, 3)# c3
        self.conv4 = ResBlock(channelnum, channelnum, 3)# c4
        self.conv5 = ResBlock(channelnum, channelnum, 3)# c5
        self.conv6 = nn.Conv2d(channelnum, 3, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv6.weight.data)
        torch.nn.init.constant_(self.conv6.bias.data, 0.0)

    def forward(self, x):
        feature_ext = self.f_relu(self.feature_ext(x))
        c0 = self.conv0(feature_ext)
        c0_p = self.conv0_p(c0)
        c1 = self.conv1(c0_p)
        c1_p = self.conv1_p(c1)
        c2 = self.conv2(c1_p)
        c3 = self.conv3(c2)
        c3_u = c1 + bilinearupsacling2(c3)#torch.nn.functional.interpolate(input=c3, scale_factor=2, mode='bilinear', align_corners=True)
        c4 = self.conv4(c3_u)
        c4_u = c0 + bilinearupsacling2(c4)# torch.nn.functional.interpolate(input=c4, scale_factor=2, mode='bilinear', align_corners=True)
        c5 = self.conv5(c4_u)
        res = self.conv6(c5)
        return res


flowfiledsSamples = [{} for i in range(8)]
class ME_Spynet(nn.Module):
    '''
    Get flow
    '''

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))# , count_include_pad=False))
            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))#, count_include_pad=False))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device_id = im1.device.index
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat([im1list[self.L - 1 - intLevel], flow_warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample), flowfiledsUpsample], 1))# residualflow

        return flowfileds



def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    # print(inputfeature.size())
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear')
    # print(outfeature.size())
    return outfeature


def bilinearupsacling2(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=True)
    return outfeature


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class Adp(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, aux_channel):
        super(Adp, self).__init__()
        self.DAB = DAB(n_feat, kernel_size, reduction, aux_channel)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_feat * 2, out_channels=n_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, mv, mv_aux2, mv_aux3):
        """
        Using mv_aux2 and mv_aux3 to adjust or strength mv.

        :param mv: main motion vector: (B * C * H * W)
        :param mv_aux2: aux motion vector: (B * C * H * W)
        :param mv_aux3: another aux motion vector: (B * C * H * W)
        :return: adjusted motion vector: (B * C * H * W)
        """
        mv_aux = self.convs(torch.cat([mv_aux2, mv_aux3], 1))
        result = self.DAB(mv, mv_aux)

        return result

class DAB_feature(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, aux_channel):
        super(DAB_feature, self).__init__()

        self.da_conv1 = feature_DA_conv(n_feat, n_feat, kernel_size, reduction, aux_channel)

        self.conv1 = default_conv(n_feat, n_feat, kernel_size)

        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1)
        self.resb=ResBlock(out_channel,out_channel)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, y):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x, y))
        out = self.conv1(out)+ x

        out2=self.conv2(out)
        out2=self.resb(out2)
        out3=self.conv2(out2)
        out3=self.resb(out3)

        return out,out2,out3

class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, aux_channel):
        super(DAB, self).__init__()
        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction, aux_channel)
        self.conv1 = default_conv(n_feat, 2, kernel_size)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))

        return out


class feature_DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction, aux_channel):
        super(feature_DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

        self.E = Encoder(aux_channel)

    def forward(self, x, y):

        b, c, h, w = x.size()

        y = self.E(y)


        # branch 1

        kernel = self.kernel(y)

        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)

        out = self.relu(F.conv2d(x.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))

        out = self.conv(out.view(b, -1, h, w))


        # branch 2
        out = out + self.ca(x, y)

        return out


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction, aux_channel):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

        self.E = Encoder(aux_channel)

    def forward(self, x):

        b, c, h, w = x.size()

        y = self.E(x)
        # branch 1
        kernel = self.kernel(y)

        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)

        out = self.relu(F.conv2d(x.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))

        out = self.conv(out.view(b, -1, h, w))


        # branch 2
        out = out + self.ca(x)

        return out+x


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 256//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256 // reduction, 64, 1, 1, 0, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        att = self.conv_du(x)

        return x * att


class Encoder(nn.Module):
    def __init__(self, aux_channel):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)
        return out




class mode_prediction_network(nn.Sequential):
    def __init__(self, in_planes: int=128, out_planes: int=128):
        super().__init__()
        self.resb1 = ResBlock(out_channel_F * 2, out_channel_F * 2)
        self.resb2 = ResBlock(out_channel_F * 2, out_channel_F * 2)
        self.resb3 = ResBlock(out_channel_F * 2, out_channel_F * 2)

        self.conv1=conv(out_planes,out_planes, kernel_size=5, stride=2)
        self.conv2=conv(out_planes,out_planes,kernel_size=3,stride=1)



    def forward(self,x):

        x1=self.resb1(x)
        x2=self.resb2(x1)
        x3=self.resb3(x2)
        x4=self.conv1(x3)

        ####Branch1
        x6=self.resb1(x4)
        x7=self.resb2(x6)
        x8=self.resb3(x7)
        x9=self.conv1(x8)
        x10=self.conv2(x9)
        x11=gumbel_softmax(x10)


        ####Branch2
        x5=self.conv2(x4)
        x12=gumbel_softmax(x5)

        return x12

def loadweightformnp(layername):
    index = layername.find('modelL')
    if index == -1:
        print('laod models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = modelspath + name + '-weight.npy'
        modelbias = modelspath + name + '-bias.npy'
        weightnp = np.load(modelweight)

        biasnp = np.load(modelbias)



        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)



class MEBasic(nn.Module):
    '''
    Get flow
    '''
    def __init__(self, layername):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv1.weight.data, self.conv1.bias.data = loadweightformnp(layername + '_F-1')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv2.weight.data, self.conv2.bias.data = loadweightformnp(layername + '_F-2')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv3.weight.data, self.conv3.bias.data = loadweightformnp(layername + '_F-3')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv4.weight.data, self.conv4.bias.data = loadweightformnp(layername + '_F-4')
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        self.conv5.weight.data, self.conv5.bias.data = loadweightformnp(layername + '_F-5')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x

flowfiledsSamples = [{} for i in range(8)]

class ME_Spynet(nn.Module):
    '''
    Get flow
    '''

    def __init__(self, layername='motion_estimation'):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList(
            [MEBasic(layername + 'modelL' + str(intLevel + 1)) for intLevel in range(4)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))
            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device_id = im1.device.index
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat(
                [im1list[self.L - 1 - intLevel], flow_warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample),
                 flowfiledsUpsample], 1))  # residualflow

        return flowfileds




############## loopfilter
class LoopFilter(nn.Module):
    def __init__(self, channels):
        super(LoopFilter, self).__init__()
        self._ch = channels

        self.conv1 = nn.Conv2d(6, self._ch, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(self._ch, self._ch) for _ in range(2)])
        self.conv2 = nn.Conv2d(self._ch, self._ch, 3, padding=1)
        self.conv3 = nn.Conv2d(self._ch, 6, 3, padding=1)

        self.conv4 = nn.Conv2d(6, 3, 3, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=1e-3)
                    nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x = inputs

        y = self.conv1(x)
        y = nn.functional.relu(y, inplace=True)

        z = self.blocks(y)

        w = self.conv2(z)
        w = nn.functional.relu(w, inplace=True)

        w = self.conv3(w)

        return self.conv4(x + w)
