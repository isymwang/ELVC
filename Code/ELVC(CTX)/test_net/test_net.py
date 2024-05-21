import argparse
import logging
from nvc import *
from fvc_net.ms_ssim_torch import *
from nvc import NVC
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from typing import List

from dataset import DataSet, UVGDataSet
from tensorboardX import SummaryWriter
torch.backends.cudnn.enabled = True
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


from dataset import *
from fvc_net.basic import cal_bpp,cal_psnr,cal_msssim

logger = logging.getLogger("VideoCompression")


def testdataset(path,net,I_codec,dataset_name):
    net.update(force=True)
    print('testing ',dataset_name)
    global test_dataset
    test_dataset = HEVCDataSet(root=path)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
    sumbpp = 0
    I_BPP=0
    P_BPP=0
    sumbpp_mv_y = 0
    sumbpp_mv_z = 0
    sumbpp_res_y = 0
    sumbpp_res_z = 0
    sumpsnr = 0
    summsssim=0
    eval_step = 0
    gop_num = 0


    for batch_idx, input in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print("testing : %d/%d"% (batch_idx, len(test_loader)))
        seqlen = input[0].size()[0]

        net.eval()
        with torch.no_grad():
            for i in range(seqlen):
                if i == 0:
                    I_frame = input[:, i, :, :, :].cuda()
                    I_frame_padding, padding_left, padding_right, padding_top, padding_bottom = imagepadding(I_frame)
                    num_pixels = 1 * I_frame.shape[-2] * I_frame.shape[-1]
                    arr = I_codec(I_frame_padding)
                    I_rec = arr['x_hat']
                    I_likelihood_y = arr["likelihoods"]['y']
                    I_likelihood_z = arr["likelihoods"]['z']

                    ref_image = I_rec.clone().detach()
                    y_bpp = cal_bpp(likelihood=I_likelihood_y, num_pixels=num_pixels).cpu().detach().numpy()
                    z_bpp = cal_bpp(likelihood=I_likelihood_z, num_pixels=num_pixels).cpu().detach().numpy()

                    I_frame_hat=F.pad(ref_image, (-padding_left, -padding_right, -padding_top, -padding_bottom))
                    I_frame_hat = I_frame_hat.clamp_(0, 1)

                    bpp_i = y_bpp + z_bpp
                    psnr_i = cal_psnr(distortion=cal_distoration(I_frame_hat, I_frame)).cpu().detach().numpy()
                    msssim_i=cal_msssim(I_frame_hat,I_frame).cpu().detach().numpy()

                    sumbpp += bpp_i
                    sumpsnr += psnr_i
                    summsssim += msssim_i
                    I_BPP+=bpp_i


                    log = "\n------------------ GOP {%d} --------------------\n" \
                          "I frame:%d  bpp:%.6f, psnr:%.6f, ms-ssim:%.6f" \
                          % (batch_idx + 1, i,bpp_i, psnr_i, msssim_i)
                    logger.info(log)

                    gop_num += 1
                    eval_step += 1

                else:
                    input_image = input[:, i, :, :, :].cuda()
                    input_image_padding, padding_left, padding_right, padding_top, padding_bottom = imagepadding(input_image)
                    torch.use_deterministic_algorithms(True)
                    torch.set_num_threads(1)

                    bpp,reconframe,mv_bpp,mv_z_bpp,res_bpp,res_z_bpp = net.test(input_image_padding, ref_image)

                    torch.use_deterministic_algorithms(False)
                    torch.set_num_threads(36)
                    ref_image = reconframe

                    reconframe_hat=F.pad(reconframe, (-padding_left, -padding_right, -padding_top, -padding_bottom))

                    reconframe_hat = reconframe_hat.clamp_(0, 1)
                    mse_loss = torch.mean((reconframe_hat - input_image).pow(2))
                    bpp_p=torch.mean(bpp).cpu().detach().numpy()
                    psnr_p=torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
                    msssim_p= ms_ssim(reconframe_hat, input_image, data_range=1.0,size_average=True).cpu().detach().numpy()

                    sumbpp += bpp_p
                    sumpsnr += psnr_p
                    summsssim += msssim_p

                    sumbpp_mv_y+=mv_bpp
                    sumbpp_mv_z+=mv_z_bpp
                    sumbpp_res_y+=res_bpp
                    sumbpp_res_z+=res_z_bpp
                    eval_step += 1
                    P_BPP+=bpp_p
                    log = "P frame:%d  bpp:%.6f, psnr:%.6f, ms-ssim:%.6f, mv_y:%.6f,mv_y_z:%.6f,context:%.6f,context_z:%.6f" \
                          % (i, bpp_p, psnr_p, msssim_p,mv_bpp,mv_z_bpp,res_bpp,res_z_bpp)
                    logger.info(log)



    sumbpp /= eval_step
    sumpsnr /= eval_step
    summsssim/= eval_step
    I_BPP/=gop_num
    P_BPP/=(eval_step - gop_num)
    sumbpp_mv_y /= (eval_step - gop_num)
    sumbpp_mv_z /= (eval_step - gop_num)
    sumbpp_res_y /= (eval_step - gop_num)
    sumbpp_res_z /= (eval_step - gop_num)

    log = "%s  : average bpp : %.6lf,I_bpp:%.6f,P_bpp:%.6f, mv_y_bpp : %.6lf, mv_z_bpp : %.6lf, res_y_bpp : %.6lf, res_z_bpp : %.6lf, average psnr : %.6lf,average_msssim: %.6lf\n" \
          % (dataset_name,sumbpp,I_BPP,P_BPP,sumbpp_mv_y, sumbpp_mv_z, sumbpp_res_y, sumbpp_res_z, sumpsnr,summsssim)
    logger.info(log)


