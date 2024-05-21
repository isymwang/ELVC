
import imageio
import os
import getopt
import math
import numpy as np
import PIL
import PIL.Image
import sys
import torch

import cv2
import scipy.misc

import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def offsets_vis(offsets,iii):
    flow_dir = "G:/wangyiming/Multi_motion_estimation_elic/visual_mv/HEVCD_RaceHorses"

    # flos = offsets[0].permute(1, 2, 0)
    flos = offsets[0]
    height, width, _ = flos.shape

    ###DCN
    for i in range(2):
        # flo = flos[:, :, 2 * i:2 * i + 2].cpu().numpy()
        flo = flos[2 * i:2 * i + 2, :, : ].cpu().numpy()
        # img_result = flowToColor(flo, maxflow=None)
        img_result = flow_to_image(flo)
        file_name = 'flo'+str(iii) +"--"+str(i + 1) + '.png'
        imageio.imwrite(os.path.join(flow_dir, file_name), img_result[0])
    print(f"Saving visualisation at {flow_dir}")





def optical_flow_vis(offsets,iii,recon_fold):
    flow_dir = "G:/wangyiming/optical_flow_constrastive_context_learning/"+recon_fold

    # flos = offsets[0].permute(1, 2, 0)
    # flos = offsets[0]
    # height, width, _ = flos.shape

    # for i in range(2):
    #     # flo = flos[:, :, 2 * i:2 * i + 2].cpu().numpy()
    #     flo = flos[2 * i:2 * i + 2, :, : ].cpu().numpy()
    #     # img_result = flowToColor(flo, maxflow=None)
    #     img_result = flow_to_image(flo)
    #     file_name = 'flo'+str(iii) +"--"+str(i + 1) + '.png'
    #     imageio.imwrite(os.path.join(flow_dir, file_name), img_result[0])
    #
    #
    # print(f"Saving visualisation at {flow_dir}")

    flow = offsets.detach().cpu().numpy()
    rgb_my = flow_to_image(flow[0, :, :, :])
    file_name = 'flo' + str(iii) + "--" + '.png'
    imageio.imwrite(os.path.join(flow_dir, file_name), rgb_my[0])
    print(f"Saving visualisation at {flow_dir,iii}")




def offsets_vis_predict(feature,iii,recon_fold):
    data_dir = "G:/wangyiming/optical_flow_context/vis_feature/predict_feature/"+recon_fold

    feature = feature[0].cpu().numpy()

    for i in range(64):


        feature_map = feature[i,:,:]


        file_name = 'feature'+str(iii) +"--"+str(i + 1) + '.png'
        imageio.imwrite(os.path.join(data_dir, file_name), feature_map)

    print(f"Saving visualisation at {data_dir}")




def vis_latent_y(feature,iii):
    import matplotlib
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt

    data_dir = "G:/wangyiming/Multi_motion_estimation_elic/visual_latent/HEVCD_RaceHorses"

    feature = feature[0].cpu().numpy()
    for i in range(10):
        plt.figure()
        ax = plt.gca()
        width = feature.shape[1]
        height = feature.shape[2]
        dpi=300
        fig = plt.figure(figsize=(width , height), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])


        im1 = ax.imshow(feature[i,:,:], cmap='coolwarm')

# divider = make_axes_locatable(ax)
#
# cax = divider.append_axes("right", size="2%", pad=0.05)
# plt.colorbar(im,cax=cax)

        ax.axis('off')

        file_name = 'feature' + str(iii) + "--" + str(i) + '.png'

        plt.savefig(os.path.join(data_dir, file_name), dpi=300, bbox_inches='tight')
    # plt.show()


def offsets_vis_recon(feature,iii,recon_fold):


    data_dir = "G:/wangyiming/optical_flow_context/vis_feature/recon_feature/" + recon_fold

    feature = feature[0].cpu().numpy()

    for i in range(64):


        feature_map = feature[i,:,:]


        file_name = 'feature'+str(iii) +"--"+str(i + 1) + '.png'
        imageio.imwrite(os.path.join(data_dir, file_name), feature_map)


def flowToColor(flow, maxflow=None, verbose=True):
    '''
    args
        flow (numpy array) height x width x 2
    return
        img_color (numpy array) height x width x 3
    '''

    # UNKNOWN_FLOW_THRESH = 5e2
    UNKNOWN_FLOW_THRESH = 1e5
    eps = 1e-6

    height, widht, nBands = flow.shape

    if nBands != 2:
        exit('flowToColor: image must have two bands')

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999
    maxrad = -1

    # fix unknown flow
    idxUnknown = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max(maxu, u.max())
    minu = min(minu, u.min())

    maxv = max(maxv, v.max())
    minv = min(minv, v.min())

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(maxrad, rad.max())

    if verbose:
        print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' %
              (maxrad, minu, maxu, minv, maxv))

    if maxflow is not None:
        if maxflow > 0:
            maxrad = maxflow

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)

    img = computeColor(u, v)

    # unknown flow
    # IDX = repmat(idxUnknown, [1, 1, 3])
    img[idxUnknown] = 0

    return img


def computeColor(u, v, cast_uint8=True):
    '''
    args
        u (numpy array) height x width
        v (numpy array) height x width
        cast_uint8 (bool) set False to have image range 0-1 (np.float32)
    return
        img_color (numpy array) height x width x 3
    '''

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = makeColorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u)/np.pi

    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols

    k0 = np.floor(fk).astype(int)  # 1, 2, ..., ncols

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1

    f = fk - k0

    height, width = u.shape
    img = np.zeros((height, width, 3), np.float32)
    nrows = colorwheel.shape[1]
    for i in range(nrows):
        tmp = colorwheel[:, i]
        col0 = tmp[k0.reshape(-1)]/255
        col1 = tmp[k1.reshape(-1)]/255
        col = col0.reshape(height, width) * (1 - f) + \
              col1.reshape(height, width) * f

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])  # increase saturation with radius

        col[np.logical_not(idx)] *= 0.75  # out of range

        img[:, :, i] = col * (1 - nanIdx)

    if cast_uint8:
        img = np.floor(img * 255).astype(np.uint8)
    return img



##########################################################
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
def makeColorwheel():
    '''
    color encoding scheme
    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.array(range(RY))/RY)
    col = col+RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.array(range(YG))/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.array(range(GC))/GC)
    col = col+GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.array(range(CB))/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.array(range(BM))/BM)
    col = col+BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.array(range(MR))/MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel




def flow_to_image(flow, display=False, maxrad = None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    u = flow[0, :, :]
    v = flow[1, :, :]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    if maxrad == None:
        maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img), maxrad


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img



def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel