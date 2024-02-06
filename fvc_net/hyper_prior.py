import torch

from fvc_net.basic import *
from fvc_net.models.google import CompressionModel, get_scale_table
from fvc_net.decoder_resblock import *
from fvc_net.encoder_resblock import *
from fvc_net.hyper_decoder import *
from fvc_net.hyper_encoder import *

from fvc_net.hyper_prior import *
from fvc_net.layers.layers import MaskedConv2d, subpel_conv3x3
from fvc_net.entropy_models import GaussianConditional,GaussianConditional_HAMC

from fvc_net.models.utils import (
    conv,
    deconv,
    quantize_ste,
    update_registered_buffers,
)
from fvc_net.layer import CheckboardMaskedConv2d

def get_downsampled_shape(height, width, p):

    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


class Hyperprior_mv(CompressionModel):
    def __init__(self, planes: int = 192, mid_planes: int = 192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        out_channel_mv = 128
        self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
        self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
        self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
        self.gaussian_conditional = GaussianConditional_HAMC(None)



    def forward(self, y):
        z = self.hyper_encoder(y)                           # y (4,128,16,16)  z(4,128,4,4)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)   # z_hat (4,128,4,4)

        scales = self.hyper_decoder_scale(z_hat)            # scales(4,128,16,16)
        means  = self.hyper_decoder_mean(z_hat)             # means(4,128,16,16)


        _, y_likelihoods = self.gaussian_conditional(y, scales, means)

        y_hat = quantize_ste(y - means) + means  # y_hat (4,128,16,16)


        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y):
        z = self.hyper_encoder(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)

        y_string = self.gaussian_conditional.compress(y, indexes, means)


        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)



        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)

        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        return y_hat

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)



class Hyperprior_res(CompressionModel):
    def __init__(self, planes: int = 192, mid_planes: int = 192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
        self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
        self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, y):
        z = self.hyper_encoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        y_hat = quantize_ste(y - means) + means

        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y):
        z = self.hyper_encoder(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)

        y_string = self.gaussian_conditional.compress(y, indexes, means)
        # y_string = self.gaussian_conditional.compress(y, indexes)


        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

        # y_hat = self.gaussian_conditional.quantize(y, "dequantize")
        # y_hat = self.gaussian_conditional.decompress(y_string, indexes, z_hat.dtype, means)

        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)

        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        return y_hat



class Hyperprior_context(CompressionModel):
    def __init__(self, planes: int = 192, mid_planes: int = 192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        out_channel_mv = 128
        self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
        self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
        self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
        self.gaussian_conditional = GaussianConditional(None)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 2, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 2, 1),
        )

    def forward(self, y,temporal_prior_params):
        z = self.hyper_encoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales = self.hyper_decoder_scale(z_hat)

        gaussian_params = self.entropy_parameters(torch.cat((temporal_prior_params, scales), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)

        y_hat = quantize_ste(y - means_hat) + means_hat
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y, temporal_prior_params):
        z = self.hyper_encoder(y)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)

        gaussian_params = self.entropy_parameters(torch.cat((temporal_prior_params, scales), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_string = self.gaussian_conditional.compress(y, indexes, means_hat)
        # y_string = self.gaussian_conditional.compress(y, indexes)

        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means_hat)

        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}


    def decompress(self, strings, shape,temporal_prior_params):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        gaussian_params = self.entropy_parameters(torch.cat((temporal_prior_params, scales), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means_hat)

        # y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        return y_hat