import time
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
from feature_map_visual.feature_visual import feature_map_vis
from flow_vis import optical_flow_vis
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


        ###motion estimation
        self.ME_Net =ME_Spynet()
        self.conv_mv=nn.Conv2d(2,out_channel_O,3,1,1)

        ###motion compensation
        self.warpnet = Warp_net()

        ###motion refinement
        self.motion_refinement = DAB(n_feat=64, kernel_size=3, reduction=2, aux_channel=64)

        ###motion compression
        self.motion_encoder = EncoderWithResblock(in_planes=2, out_planes=out_channel_M)
        self.motion_decoder = DecoderWithResblock(in_planes=out_channel_M, out_planes=2)

        self.motion_hyperprior = Hyperprior_mv(planes=out_channel_M, mid_planes=out_channel_M)


        ###residual
        # self.resEncoder = Analysis_net(in_planes=out_channel_F, out_planes=out_channel_F)
        # self.resDecoder = Synthesis_net(in_planes=out_channel_F, out_planes=out_channel_F)
        # self.res_hyperprior = Hyperprior_res(planes=128, mid_planes=128)



        ##context

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )
        self.context_hyperprior = Hyperprior_context(planes=128, mid_planes=128)

        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )


        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N + 3, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),

            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),

            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N * 2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.contextualDecoder_refinement = nn.Sequential(
            nn.Conv2d(out_channel_N * 2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),

        )

        self.loop_filter = LoopFilter(out_channel_M//2)

    def motioncompensation(self, ref, mv):
        ref_feature = self.feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context = self.context_refine(prediction_init)

        return context

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


    def forward(self,input_image, referframe):
        mv = self.ME_Net(input_image, referframe)
        mv_feature=self.conv_mv(mv)
        mv_refine=self.motion_refinement(mv_feature)
        ###motion compression
        # encode motion info
        mv_feature = self.motion_encoder(mv_refine)
        quant_mv, motion_likelihoods = self.motion_hyperprior(mv_feature)
        # decode motion info
        mv_hat= self.motion_decoder(quant_mv)

        # motion compensation
        prediction=self.motioncompensation(referframe, mv_hat)


        ###residual coding
        # input_residual = input_image - prediction
        # feature = self.resEncoder(input_residual)
        # quant_res, res_likelihoods = self.res_hyperprior(feature)
        # res_hat = self.resDecoder(quant_res)
        # recon_image=res_hat+prediction


        ###context coding
        temporal_prior_params = self.temporalPriorEncoder(prediction)

        encoded_context = self.contextualEncoder(torch.cat((input_image, prediction), dim=1))

        context_hat, context_likelihoods = self.context_hyperprior(encoded_context, temporal_prior_params)

        recon_image = self.contextualDecoder_part1(context_hat)

        recon_image = self.contextualDecoder_part2(torch.cat((recon_image, prediction), dim=1))

        x_hat = self.loop_filter(torch.cat([referframe, recon_image], 1)) + recon_image

        batch_size = context_hat.size()[0]

        clipped_recon_image = x_hat.clamp(0., 1.)

        ####Rate_LOSS
        mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))
        im_shape = input_image.size()
        bpp_mv = torch.log(motion_likelihoods['y']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_mvprior = torch.log(motion_likelihoods['z']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_context = torch.log(context_likelihoods['y']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])
        bpp_contextprior = torch.log(context_likelihoods['z']).sum() / (-math.log(2) * batch_size * im_shape[2] * im_shape[3])

        bpp = bpp_mv + bpp_mvprior + bpp_context+ bpp_contextprior
        return clipped_recon_image, mse_loss,  bpp_context, bpp_contextprior, bpp_mv, bpp_mvprior, bpp


    def compress(self,input_image, referframe,frame_i):
        mv = self.ME_Net(input_image, referframe)

        # optical_flow_vis(mv,frame_i,'coarse_MV')

        mv_feature = self.conv_mv(mv)
        mv_refine = self.motion_refinement(mv_feature)
        # optical_flow_vis(mv_refine, frame_i, 'refine_MV')

        ###motion compression
        # encode motion info
        mv_feature = self.motion_encoder(mv_refine)

        quant_mv, out_motion = self.motion_hyperprior.compress(mv_feature)

        # decode motion info
        mv_hat = self.motion_decoder(quant_mv)

        # motion compensation
        prediction=self.motioncompensation(referframe, mv_hat)


        ###residual coding
        # input_residual = input_image - prediction
        #
        # feature = self.resEncoder(input_residual)
        # quant_res, out_res = self.res_hyperprior.compress(feature)
        # res_hat = self.resDecoder(quant_res)

        ###contextual coding
        temporal_prior_params = self.temporalPriorEncoder(prediction)

        encoded_context = self.contextualEncoder(torch.cat((input_image, prediction), dim=1))

        context_hat, out_context = self.context_hyperprior.compress(encoded_context, temporal_prior_params)

        recon_image = self.contextualDecoder_part1(context_hat)

        recon_image = self.contextualDecoder_part2(torch.cat((recon_image, prediction), dim=1))

        x_hat = self.loop_filter(torch.cat([referframe, recon_image], 1)) + recon_image



        return {
            "strings": {
                "motion": out_motion["strings"],
                "context": out_context["strings"],
            },
            "shape": {
                "motion": out_motion["shape"],
                "context": out_context["shape"]},
        }


    def decompress(self,referframe,strings,shapes):


        # motion
        key = "motion"
        quant_mv = self.motion_hyperprior.decompress(strings[key], shapes[key])
        mv_hat = self.motion_decoder(quant_mv)


        # motion compensation

        prediction=self.motioncompensation(referframe, mv_hat)


        temporal_prior_params = self.temporalPriorEncoder(prediction)

        # res
        key = "context"

        context_hat= self.context_hyperprior.decompress(strings[key], shapes[key],temporal_prior_params)


        recon_image = self.contextualDecoder_part1(context_hat)

        recon_image = self.contextualDecoder_part2(torch.cat((recon_image, prediction), dim=1))

        x_hat = self.loop_filter(torch.cat([referframe, recon_image], 1)) + recon_image


        x_rec= x_hat.clamp(0., 1.)
        return {"x_hat": x_rec}



    def load_state_dict(self, state_dict):

        # Dynamically update the entropy bottleneck buffers related to the CDFs

        update_registered_buffers(
            self.context_hyperprior.gaussian_conditional,
            "context_hyperprior.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.context_hyperprior.entropy_bottleneck,
            "context_hyperprior.entropy_bottleneck",
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

        updated = self.context_hyperprior.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.context_hyperprior.entropy_bottleneck.update(force=force)

        updated |= self.motion_hyperprior.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.motion_hyperprior.entropy_bottleneck.update(force=force)

        return updated


    def test(self,input_image, ref_image,frame_i):


        strings_and_shape = self.compress(input_image, ref_image,frame_i)


        strings, shape = strings_and_shape["strings"], strings_and_shape["shape"]

        reconframe = self.decompress(ref_image, strings, shape)["x_hat"]




        num_pixels = input_image.size()[2] * input_image.size()[3]
        num_pixels = torch.tensor(num_pixels).float()
        mv_y_string=strings["motion"][0][0]
        mv_z_string=strings["motion"][1][0]
        context_y_string=strings["context"][0][0]
        context_z_string=strings["context"][1][0]
        mv_bpp=len(mv_y_string)* 8.0 / num_pixels
        mv_z_bpp=len(mv_z_string)* 8.0 / num_pixels

        context_bpp=len( context_y_string)* 8.0 / num_pixels
        context_z_bpp=len( context_z_string)* 8.0 / num_pixels


        bpp = mv_bpp+mv_z_bpp+ context_bpp+ context_z_bpp

        reconframe = reconframe.clamp(0., 1.)


        # return bpp,reconframe,mv_bpp,mv_z_bpp, context_bpp, context_z_bpp
        return bpp,reconframe,mv_bpp,mv_z_bpp, context_bpp, context_z_bpp











