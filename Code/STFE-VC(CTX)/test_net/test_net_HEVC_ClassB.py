import argparse
import logging
import time
from nvc import *
from fvc_net.ms_ssim_torch import *
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from typing import List
from nvc import NVC
from nvc_ssim import NVC_ssim
from dataset import DataSet, UVGDataSet
from tensorboardX import SummaryWriter
from thop import profile
torch.backends.cudnn.enabled = True
from fvc_net.RDloss import *

import os
from dataset import *
from fvc_net.zoo.image import cheng2020_anchor

torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()


print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0#  // gpu_num
gpu_per_batch = 4
test_step = 10000#  // gpu_num
tot_epoch = 1000000
tot_step = 2000000

logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0

# ref_i_dir = geti(train_lambda)


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

parser = argparse.ArgumentParser(description='FVC reimplement')

parser.add_argument('-l', '--log', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument("--lambda_weight", type=int, default=2048, help="the lambda value")
parser.add_argument('--testhevcB', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', required=True,
        help = 'hyperparameter of Reid in json format')
parser.add_argument('--msssim', action='store_true')

def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']


hevc_b_path='G:/wangyiming/data/test_data/HEVC_ClassB/images'

lambda_I_quality_map = {8:3,
                        16:4,
                        32:5,
                        64:6,
                        128:6,
                        256: 3,
                        512: 4,
                        1024: 5,
                        2048: 6}




def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0)*255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype('uint8'), 'RGB')
    frame_result.save(path)

recon_bin_folder = os.path.join('recon_frame', 'hevc_classB', 'BasketballDrive')

if not os.path.exists(recon_bin_folder):
    os.makedirs(recon_bin_folder)


def testhevc(path,net,I_codec,dataset_name):
    net.update(force=True)
    print('testing ', dataset_name)
    global test_dataset
    test_dataset = HEVCBDataSet(root=path)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
    sumbpp = 0
    I_BPP = 0
    sumbpp_mv_y = 0
    sumbpp_mv_z = 0
    sumbpp_res_y = 0
    sumbpp_res_z = 0
    sumpsnr = 0
    summsssim = 0
    eval_step = 0
    gop_num = 0
    ii=0
    for batch_idx, input in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print("testing : %d/%d" % (batch_idx, len(test_loader)))
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

                    I_frame_hat = F.pad(ref_image, (-padding_left, -padding_right, -padding_top, -padding_bottom))
                    I_frame_hat = I_frame_hat.clamp_(0, 1)

                    bpp_i = y_bpp + z_bpp
                    psnr_i = cal_psnr(distortion=cal_distoration(I_frame_hat, I_frame)).cpu().detach().numpy()
                    msssim_i = cal_msssim(I_frame_hat, I_frame).cpu().detach().numpy()
                    # write_torch_frame(I_frame_hat.squeeze(),
                    #                   os.path.join(recon_bin_folder, f"recon_frame_{ii * 10 + i}.png"))
                    sumbpp += bpp_i
                    sumpsnr += psnr_i
                    summsssim += msssim_i
                    I_BPP += bpp_i

                    log = "\n------------------ GOP {%d} --------------------\n" \
                          "I frame:%d  bpp:%.6f, psnr:%.6f, ms-ssim:%.6f" \
                          % (batch_idx + 1, i, bpp_i, psnr_i, msssim_i)
                    logger.info(log)

                    gop_num += 1
                    eval_step += 1

                else:
                    input_image = input[:, i, :, :, :].cuda()
                    input_image_padding, padding_left, padding_right, padding_top, padding_bottom = imagepadding(input_image)

                    torch.use_deterministic_algorithms(True)
                    torch.set_num_threads(1)

                    bpp, reconframe, mv_bpp, mv_z_bpp, res_bpp, res_z_bpp = net.test(input_image_padding, ref_image,eval_step)



                    torch.use_deterministic_algorithms(False)
                    torch.set_num_threads(36)
                    ref_image = reconframe



                    reconframe_hat = F.pad(reconframe, (-padding_left, -padding_right, -padding_top, -padding_bottom))



                    reconframe_hat = reconframe_hat.clamp_(0, 1)
                    mse_loss = torch.mean((reconframe_hat - input_image).pow(2))



                    bpp_p = torch.mean(bpp).cpu().detach().numpy()
                    psnr_p = torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()

                    msssim_p = ms_ssim(reconframe_hat, input_image, data_range=1.0,size_average=True).cpu().detach().numpy()

                    # write_torch_frame(reconframe_hat.squeeze(),os.path.join(recon_bin_folder, f"recon_frame_{ii * 10 + i}.png"))

                    sumbpp += bpp_p
                    sumpsnr += psnr_p
                    summsssim += msssim_p

                    sumbpp_mv_y += mv_bpp
                    sumbpp_mv_z += mv_z_bpp
                    sumbpp_res_y += res_bpp
                    sumbpp_res_z += res_z_bpp
                    eval_step += 1
                    log = "P frame:%d  bpp:%.6f, psnr:%.6f, ms-ssim:%.6f, mv_y:%.6f,mv_y_z:%.6f,context:%.6f,context_z:%.6f" \
                          % (i, bpp_p, psnr_p, msssim_p, mv_bpp, mv_z_bpp, res_bpp, res_z_bpp)
                    logger.info(log)

        ii+=1

    sumbpp /= eval_step
    sumpsnr /= eval_step
    summsssim /= eval_step
    I_BPP /= gop_num
    sumbpp_mv_y /= (eval_step - gop_num)
    sumbpp_mv_z /= (eval_step - gop_num)
    sumbpp_res_y /= (eval_step - gop_num)
    sumbpp_res_z /= (eval_step - gop_num)

    log = "%s  : average bpp : %.6lf,I_bpp:%.6f, mv_y_bpp : %.6lf, mv_z_bpp : %.6lf, res_y_bpp : %.6lf, res_z_bpp : %.6lf, average psnr : %.6lf,average_msssim: %.6lf\n" \
          % (dataset_name, sumbpp, I_BPP, sumbpp_mv_y, sumbpp_mv_z, sumbpp_res_y, sumbpp_res_z, sumpsnr, summsssim)
    logger.info(log)


if __name__ == "__main__":

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(seed=0)

    args = parser.parse_args()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.log != '':
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("FVC training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)

    model = NVC()
    model_ssim=NVC_ssim()
    net = model.cuda()
    net_ssim=model_ssim.cuda()

    I_codec = cheng2020_anchor(quality=lambda_I_quality_map[train_lambda], metric='mse', pretrained=True).cuda()

    input_shape = (3, 1920, 1088)
    input_tensor = torch.randn(1, *input_shape).cuda()
    flops, params = profile(model, inputs=(input_tensor, input_tensor))
    print('FLOPs = ' + str(flops / 1000 ** 4) + 'T')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


    I_codec.eval()
    I_codec_msssim.eval()

    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)

    if args.msssim:
        testhevc(hevc_b_path, net, I_codec_msssim, 'HEVC_Class_B_MSSSIM')

    else:
        testhevc(hevc_b_path,net,I_codec,'HEVC_Class_B')

    exit(0)




