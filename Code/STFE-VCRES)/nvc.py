from typing import List

import torch
import torch.nn as nn
from fvc_net.models.utils import (conv,deconv,quantize_ste,update_registered_buffers,)

from fvc_net.feature import *
from fvc_net.motion import *
from fvc_net.decoder_resblock import *
from fvc_net.encoder_resblock import *
from fvc_net.hyper_decoder import *
from fvc_net.hyper_encoder import *
from fvc_net.hyper_prior import *
from fvc_net.layers.layers import GDN, MaskedConv2d
from fvc_net.layer import *
from fvc_net.multi_context import *
from mmcv.ops import ModulatedDeformConv2d as DCN


import dataset

def save_model(model,iter,model_save, train_lambda,stages):
    torch.save(model.state_dict(), "{}/{}/{}-stage/iter{}.model".format(model_save,train_lambda,stages,iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0



class NVC(nn.Module):
    def __init__(self,):
        super().__init__()
        self.out_channel_mv=128
        self.out_channel_F = out_channel_F
        self.out_channel_O = out_channel_O
        self.out_channel_M = out_channel_M



        self.ME_Net =ME_Spynet()
        self.conv_mv=nn.Conv2d(2,out_channel_O,3,1,1)


        self.warpnet = Warp_net()


        self.motion_refinement = DAB(n_feat=64, kernel_size=3, reduction=2, aux_channel=64)


        self.motion_encoder = EncoderWithResblock(in_planes=2, out_planes=out_channel_M)
        self.motion_decoder = DecoderWithResblock(in_planes=out_channel_M, out_planes=2)

        self.motion_hyperprior = Hyperprior_mv(planes=out_channel_M, mid_planes=out_channel_M)



        self.resEncoder = Analysis_net(in_planes=out_channel_F, out_planes=out_channel_F)
        self.resDecoder = Synthesis_net(in_planes=out_channel_F, out_planes=out_channel_F)
        self.res_hyperprior = Hyperprior_res(planes=128, mid_planes=128)

        self.loop_filter = LoopFilter(out_channel_M//2)

    def MC_net(self,ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction



    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list



    def compress(self,input_image, referframe,frame_i):
        mv = self.ME_Net(input_image, referframe)

        mv_feature = self.conv_mv(mv)
        mv_refine = self.motion_refinement(mv_feature)
        mv_feature = self.motion_encoder(mv_refine)
        quant_mv, out_motion = self.motion_hyperprior.compress(mv_feature)

        mv_hat = self.motion_decoder(quant_mv)

        prediction = self.MC_net(referframe, mv_hat)

        input_residual = input_image - prediction

        feature = self.resEncoder(input_residual)
        quant_res, out_res = self.res_hyperprior.compress(feature)
        res_hat = self.resDecoder(quant_res)

        recon_image = res_hat + prediction

        x_hat = self.loop_filter(torch.cat([referframe, recon_image], 1)) + recon_image


        return {
            "strings": {
                "motion": out_motion["strings"],
                "res": out_res["strings"],
            },
            "shape": {
                "motion": out_motion["shape"],
                "res": out_res["shape"]},
        }


    def decompress(self,referframe,strings,shapes):

        key = "motion"
        quant_mv = self.motion_hyperprior.decompress(strings[key], shapes[key])
        mv_hat = self.motion_decoder(quant_mv)

        prediction = self.MC_net(referframe, mv_hat)

        key = "res"

        quant_res= self.res_hyperprior.decompress(strings[key], shapes[key])
        res_hat = self.resDecoder(quant_res)

        recon_image = res_hat + prediction

        x_hat = self.loop_filter(torch.cat([referframe, recon_image], 1)) + recon_image

        x_rec= x_hat.clamp(0., 1.)
        return {"x_hat": x_rec}


    def load_state_dict(self, state_dict):

        # Dynamically update the entropy bottleneck buffers related to the CDFs

        update_registered_buffers(
            self.res_hyperprior.gaussian_conditional,
            "res_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.res_hyperprior.entropy_bottleneck,
            "res_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        update_registered_buffers(
            self.motion_hyperprior.gaussian_conditional,
            "motion_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.motion_hyperprior.entropy_bottleneck,
            "motion_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )

        super().load_state_dict(state_dict)


    def update(self, scale_table=None, force=False):

        SCALES_MIN = 0.11
        SCALES_MAX = 256
        SCALES_LEVELS = 64

        def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
            return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

        if scale_table is None:
            scale_table = get_scale_table()
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.res_hyperprior.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.res_hyperprior.entropy_bottleneck.update(force=force)

        updated |= self.motion_hyperprior.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.motion_hyperprior.entropy_bottleneck.update(force=force)

        return updated


    def test(self,input_image, ref_image,frame_i):

        strings_and_shape = self.compress(input_image, ref_image, frame_i)

        strings, shape = strings_and_shape["strings"], strings_and_shape["shape"]

        reconframe = self.decompress(ref_image, strings, shape)["x_hat"]

        num_pixels = input_image.size()[2] * input_image.size()[3]
        num_pixels = torch.tensor(num_pixels).float()
        mv_y_string=strings["motion"][0][0]
        mv_z_string=strings["motion"][1][0]
        res_y_string=strings["res"][0][0]
        res_z_string=strings["res"][1][0]
        mv_bpp=len(mv_y_string)* 8.0 / num_pixels
        mv_z_bpp=len(mv_z_string)* 8.0 / num_pixels

        res_bpp=len(res_y_string)* 8.0 / num_pixels
        res_z_bpp=len(res_z_string)* 8.0 / num_pixels


        bpp = mv_bpp+mv_z_bpp+res_bpp+res_z_bpp

        reconframe = reconframe.clamp(0., 1.)

        return bpp,reconframe,mv_bpp,mv_z_bpp,res_bpp,res_z_bpp











