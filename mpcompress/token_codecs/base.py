import torch
from compressai.models.base import CompressionModel
from mpcompress.utils.coder import encode_uniform_to_bits, decode_uniform_from_bits


class UniformTokenCodec(CompressionModel):
    def __init__(self, alphabet_size, **kwargs):
        super().__init__()
        self.alphabet_size = alphabet_size

    def forward(self, tokens):
        return {
            "likelihoods": {"y": self._uniform_likelihood(tokens)},
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
            "strings": [string],
            "shape": tuple(tokens.shape),
        }

    def decompress(self, strings, shape, **kwargs):
        symbols_len = 1
        for dim in shape:  # no batch dim
            symbols_len *= dim
        alphabet_size = self.alphabet_size
        tokens = decode_uniform_from_bits(strings[0], symbols_len, alphabet_size)
        tokens = tokens.reshape(shape).long().cuda()
        return {"tokens": tokens}

