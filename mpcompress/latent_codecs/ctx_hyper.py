# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Dict, List, Tuple, Mapping

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.ops import quantize_ste
from compressai.registry import register_module

from compressai.latent_codecs.base import LatentCodec

__all__ = [
    "HyperLatentCodecWithCtx",
    "HyperpriorLatentCodecWithCtx",
]


@register_module("HyperLatentCodecWithCtx")
class HyperLatentCodecWithCtx(LatentCodec):
    """Entropy bottleneck codec with surrounding `h_a` and `h_s` transforms.

    "Hyper" side-information branch introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: ``HyperLatentCodec`` should be used inside
       ``HyperpriorLatentCodec`` to construct a full hyperprior.

    .. code-block:: none

               ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
        y ──►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►── params
               └───┘     └───┘        EB        └───┘

    """

    def __init__(
        self,
        entropy_bottleneck: EntropyBottleneck,
        h_a: nn.Module,
        h_s: nn.Module,
        quantizer: str = "noise",
        **kwargs,
    ):
        super().__init__()
        self.entropy_bottleneck = entropy_bottleneck
        self.h_a = h_a
        self.h_s = h_s
        self.quantizer = quantizer

    def forward(self, y: Tensor, ctx: Tensor) -> Dict[str, Any]:
        z = self.h_a(y, ctx)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.quantizer == "ste":
            z_medians = self.entropy_bottleneck._get_medians()
            z_hat = quantize_ste(z - z_medians) + z_medians
        params = self.h_s(z_hat, ctx)
        return {"likelihoods": {"z": z_likelihoods}, "params": params}

    def compress(self, y: Tensor, ctx: Tensor) -> Dict[str, Any]:
        z = self.h_a(y, ctx)
        shape = z.size()[-2:]
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s(z_hat, ctx)
        return {"strings": [z_strings], "shape": shape, "params": params}

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int], ctx: Tensor, **kwargs
    ) -> Dict[str, Any]:
        (z_strings,) = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s(z_hat, ctx)
        return {"params": params}


@register_module("HyperpriorLatentCodecWithCtx")
class HyperpriorLatentCodecWithCtx(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` from ``hyper`` branch.

    Hyperprior entropy modeling introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. code-block:: none

                 ┌──────────┐
            ┌─►──┤ lc_hyper ├──►─┐
            │    └──────────┘    │
            │                    ▼ params
            │                    │
            │                 ┌──┴───┐
        y ──┴───────►─────────┤ lc_y ├───►── y_hat
                              └──────┘

    By default, the following codec is constructed:

    .. code-block:: none

                 ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            ┌─►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►─┐
            │    └───┘     └───┘        EB        └───┘    │
            │                                              │
            │                  ┌──────────────◄────────────┘
            │                  │            params
            │               ┌──┴──┐
            │               │  EP │
            │               └──┬──┘
            │                  │
            │   ┌───┐  y_hat   ▼
        y ──┴─►─┤ Q ├────►────····────►── y_hat
                └───┘          GC

    Common configurations of latent codecs include:
     - entropy bottleneck ``hyper`` (default) and gaussian conditional ``y`` (default)
     - entropy bottleneck ``hyper`` (default) and autoregressive ``y``
    """

    def __init__(self, latent_codec: Mapping[str, LatentCodec], **kwargs):
        super().__init__()
        self.y = latent_codec["y"]
        self.hyper = latent_codec["hyper"]
        self.latent_codec = latent_codec

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor, ctx: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"](y, ctx)
        y_out = self.latent_codec["y"](y, hyper_out["params"])
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
            "y_hat": y_out["y_hat"],
        }

    def compress(self, y: Tensor, ctx: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"].compress(y, ctx)
        y_out = self.latent_codec["y"].compress(y, hyper_out["params"])
        [z_strings] = hyper_out["strings"]
        return {
            "strings": [*y_out["strings"], z_strings],
            "shape": {"y": y_out["shape"], "hyper": hyper_out["shape"]},
            "y_hat": y_out["y_hat"],
        }

    def decompress(
        self, strings: List[List[bytes]], shape: Dict[str, Tuple[int, ...]], ctx: Tensor, **kwargs
    ) -> Dict[str, Any]:
        *y_strings_, z_strings = strings
        assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec["hyper"].decompress([z_strings], shape["hyper"], ctx)
        y_out = self.latent_codec["y"].decompress(
            y_strings_, shape["y"], hyper_out["params"]
        )
        return {"y_hat": y_out["y_hat"]}
