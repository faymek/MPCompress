import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)


def ste_round(x):
    return torch.round(x) - x.detach() + x


def conv(in_channels, out_channels, kernel_size=5, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class HyperVitTokenCodec(CompressionModel):
    def __init__(self, feat_dims, N=1024, M=512):
        """
        input feature: ViT features with downsampling rate=16
        N: embedding dims
        M: bottleneck dims
        """
        super().__init__()

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(feat_dims, N, 1),
            ResidualBlockWithStride(N, N, 1),
            ResidualBlockWithStride(N, N, 1),
            ResidualBlockWithStride(N, M, 1),
        )
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M, N, 1),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlockUpsample(N, feat_dims, 1),
        )
        self.entropy_bottleneck = EntropyBottleneck(384)  #
        self.gaussian_conditional = GaussianConditional(None)
        self.h_a = nn.Sequential(
            ResidualBlockWithStride(M, N, 2),
            ResidualBlockWithStride(N, N, 1),
            ResidualBlockWithStride(N, 384, 2),
        )
        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample(384, N, 2),
            ResidualBlockUpsample(N, N, 2),
            conv(N, N // 2, stride=1, kernel_size=3),
            nn.GELU(),
            conv(N // 2, M, stride=1, kernel_size=3),
        )
        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample(384, N, 2),
            ResidualBlockUpsample(N, N, 2),
            conv(N, N // 2, stride=1, kernel_size=3),
            nn.GELU(),
            conv(N // 2, M, stride=1, kernel_size=3),
        )
        self.y_bpp = 0.0
        self.z_bpp = 0.0

    def forward(self, x, target=None):
        # self.gaussian_conditional.eval()
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3] * 256
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        _, y_likelihood = self.gaussian_conditional(y, latent_scales, latent_means)
        y_hat = ste_round(y - latent_means) + latent_means
        x_hat = self.g_s(y_hat)

        y_bpp = torch.log(y_likelihood).sum() / (-math.log(2) * num_pixels)
        z_bpp = torch.log(z_likelihood).sum() / (-math.log(2) * num_pixels)
        mse = torch.nn.functional.mse_loss(x_hat, x.detach())
        return y_bpp + z_bpp, mse, x_hat

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scale = self.h_scale_s(z_hat)
        mean = self.h_mean_s(z_hat)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        index = self.gaussian_conditional.build_indexes(scale)
        y_q = self.gaussian_conditional.quantize(y, "symbols", mean)
        y_hat = y_q + mean

        symbols_list.extend(y_q.reshape(-1).tolist())
        indexes_list.extend(index.reshape(-1).tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales = self.h_scale_s(z_hat)
        means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        index = self.gaussian_conditional.build_indexes(scales)
        rv = decoder.decode_stream(
            index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
        y_hat = self.gaussian_conditional.dequantize(rv, means)
        x_hat = self.g_s(y_hat)  # .clamp_(0, 1)
        return {"x_hat": x_hat}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)
