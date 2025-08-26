import torch
from compressai.models.base import CompressionModel
from mpcompress.utils.coder import encode_uniform_to_bits, decode_uniform_from_bits


class UniformTokenCodec(CompressionModel):
    def __init__(self, alphabet_size, **kwargs):
        super().__init__()
        self.alphabet_size = alphabet_size

    def forward(self, tokens):
        return {
            "likelihoods": {"t": self._uniform_likelihood(tokens)},
            "tokens": tokens,
        }

    def _uniform_likelihood(self, tokens):
        likelihoods = torch.ones(tokens.shape) * (1.0 / self.alphabet_size)
        likelihoods = likelihoods.to(tokens.device)
        return likelihoods

    def compress(self, tokens):  # no batch dim
        alphabet_size = self.alphabet_size
        string = encode_uniform_to_bits(tokens.flatten(), alphabet_size)
        return {
            "strings": {"t": [[string]]},  # for consistent API
            "shape": {"t": tuple(tokens.shape)},
        }

    def decompress(self, strings, shape, **kwargs):
        # for consistent API
        _strings = strings["t"][0][0]
        _shape = shape["t"]
        symbols_len = 1
        for dim in _shape:  # no batch dim
            symbols_len *= dim
        alphabet_size = self.alphabet_size
        tokens = decode_uniform_from_bits(_strings, symbols_len, alphabet_size)
        tokens = tokens.reshape(_shape).long().cuda()
        return {"tokens": tokens}

