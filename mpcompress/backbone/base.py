import torch
import torch.nn as nn

from einops import rearrange
import timm
import torchvision.transforms as transforms
from mpcompress.backbone.vqgan.vq_model import VQModel  # type: ignore


class VqganBackbone(nn.Module):
    def __init__(self, vqgan_config, **kwargs):
        super().__init__()
        self.vqgan = VQModel(**vqgan_config)
        self.codebook_size = self.vqgan.quantize.embedding.weight.size()[0]

    def encode(self, x):
        # x: (B, 3, H, W), (0, 1) range
        # note that: original VQGAN accept (-1,1) range
        # note that: z_q' = z + (z_q - z).detach()
        # this incurs a small MSE(z_q', z_q) = 1e-18
        # so we use z_q as the context
        z = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
        z_q_prime, _, (_, _, idxs_1d) = self.vqgan.quantize(z)
        z_shape = z_q_prime.shape[-2:]
        tokens = idxs_1d.reshape(z_shape[0], z_shape[1])
        z_q = self.vqgan.quantize.embedding(idxs_1d)
        z_q = rearrange(z_q, "(H W) C -> 1 C H W", H=z_shape[0], W=z_shape[1])
        return {"z": z, "z_q": z_q, "tokens": tokens, "shape": z_shape}

    def tokens_to_features(self, tokens):
        z_q = self.vqgan.quantize.embedding(tokens.flatten())
        z_q = rearrange(z_q, "(H W) C -> 1 C H W", H=tokens.shape[0], W=tokens.shape[1])
        return z_q

    def decode(self, z_q):
        x_hat = self.vqgan.decode(z_q)
        x_hat = (x_hat + 1) / 2
        return x_hat


class Dinov2TimmBackbone(nn.Module):
    # dinov2 of timm implementation, support varing patch size and dynamic image size
    def __init__(
        self,
        model_size="small",
        img_size=256,
        patch_size=16,
        dynamic_size=False,
        slot=-4,  # cut position, -4 means the last 4th block
        n_last_blocks=4,  # number of last blocks to take
        ckpt_path=None,
    ):
        super().__init__()
        self.n_last_blocks = n_last_blocks
        assert model_size in ["small", "base", "large", "giant"]
        self.model_size = model_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.dynamic_size = dynamic_size
        self.slot = slot
        self.n_last_blocks = n_last_blocks
        self.ckpt_path = ckpt_path
        self.model = self.load_timm_model()
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )

    def load_timm_model(self):
        feature_model = timm.create_model(
            f"vit_{self.model_size}_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=self.img_size,
            patch_size=self.patch_size,
            drop_path_rate=0.0,
            dynamic_img_size=self.dynamic_size,
        )
        feature_model.eval()
        return feature_model

    def forward(self, x, task="whole"):
        assert task in ["whole", "cls", "seg"]
        with torch.inference_mode():
            h = self.encode(x, self.slot)
            token_res = (x.size(2) // self.patch_size, x.size(3) // self.patch_size)
            h = self.decode(h, token_res=token_res, task=task)
            return h

    def encode(self, x):
        dino = self.model
        x = self.input_transform(x)
        x = dino.patch_embed(x)
        x = dino._pos_embed(x)
        x = dino.patch_drop(x)
        x = dino.norm_pre(x)
        for i, blk in enumerate(dino.blocks[: self.slot]):
            x = blk(x)
        return x

    def decode(self, h, token_res=None, task="whole"):
        if task == "whole":
            return self.decode_whole(h, token_res=token_res)
        elif task == "cls":
            return self.decode_cls(h, token_res=token_res)
        elif task == "seg":
            return self.decode_seg(h, token_res=token_res)

    def _decode(
        self,
        x,
        slot=-4,
        n=4,  # Layers or n last layers to take
        norm=True,
        return_format="[whole]",
        token_res=None,
    ):
        allow_formats = ["[whole]", "[cls,patch]", "[cls]", "[patch]", "[patch2d]"]
        assert return_format in allow_formats, (
            f"return_format must be one of {allow_formats}"
        )
        dino = self.model

        # If n is an int, take the n last blocks. If it's a list, take them
        multi_outputs = []
        if slot is None:
            multi_outputs.append(x)  # x is just the last layer
        else:
            total_block_len = len(dino.blocks)
            if isinstance(n, int):
                blocks_to_take = range(total_block_len - n, total_block_len)
            else:
                blocks_to_take = n

            for i, blk in enumerate(dino.blocks[slot:]):
                x = blk(x)
                block_num = total_block_len + slot + i
                if block_num in blocks_to_take:
                    multi_outputs.append(x)
            assert len(multi_outputs) == len(blocks_to_take), (
                f"only {len(multi_outputs)} / {len(blocks_to_take)} blocks found"
            )

        if norm:
            multi_outputs = [dino.norm(out) for out in multi_outputs]

        if return_format == "[whole]":
            return multi_outputs

        multi_class_tokens = [out[:, 0] for out in multi_outputs]
        multi_patch_tokens = [out[:, dino.num_prefix_tokens :] for out in multi_outputs]

        if return_format == "[cls,patch]":
            # feature list: [[cls token, patch tokens], ..., [cls token, patch tokens]]
            return tuple(zip(multi_class_tokens, multi_patch_tokens))
        elif return_format == "[cls]":
            return multi_class_tokens
        elif return_format == "[patch]":
            return multi_patch_tokens
        elif return_format == "[patch2d]":
            h, w = token_res
            multi_patch_tokens = [
                rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
                for out in multi_patch_tokens
            ]
            return multi_patch_tokens

    def decode_whole(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[whole]",
            token_res=token_res,
        )

    def decode_cls(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[cls,patch]",
            token_res=token_res,
        )

    def decode_seg(self, h, token_res):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[patch2d]",
            token_res=token_res,
        )


class VqganBackbone(nn.Module):
    def __init__(self, vqgan_config, **kwargs):
        super().__init__()
        self.vqgan = VQModel(**vqgan_config)
        self.codebook_size = self.vqgan.quantize.embedding.weight.size()[0]

    def encode(self, x):
        # x: (B, 3, H, W), (0, 1) range
        # note that: original VQGAN accept (-1,1) range
        # note that: z_q' = z + (z_q - z).detach()
        # this incurs a small MSE(z_q', z_q) = 1e-18
        # so we use z_q as the context
        z = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
        z_q_prime, _, (_, _, idxs_1d) = self.vqgan.quantize(z)
        z_shape = z_q_prime.shape[-2:]
        tokens = idxs_1d.reshape(z_shape[0], z_shape[1])
        z_q = self.vqgan.quantize.embedding(idxs_1d)
        z_q = rearrange(z_q, "(H W) C -> 1 C H W", H=z_shape[0], W=z_shape[1])
        return {"z": z, "z_q": z_q, "tokens": tokens, "shape": z_shape}

    def tokens_to_features(self, tokens):
        z_q = self.vqgan.quantize.embedding(tokens.flatten())
        z_q = rearrange(z_q, "(H W) C -> 1 C H W", H=tokens.shape[0], W=tokens.shape[1])
        return z_q

    def decode(self, z_q):
        x_hat = self.vqgan.decode(z_q)
        x_hat = (x_hat + 1) / 2
        return x_hat
