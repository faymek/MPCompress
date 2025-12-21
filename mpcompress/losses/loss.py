import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleLoss(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def forward(self, output, x):
        loss = output["loss"]
        monitor = {
            "loss": loss.detach().mean().item(),
        }
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


class MPC2Loss(nn.Module):
    def __init__(
        self,
        rlmbda=24.0,
    ):
        super().__init__()
        self.rlmbda = rlmbda

    def get_rlmbda(self, global_step=None):
        return self.rlmbda

    def forward(self, output, x, x_shape=None, global_step=None):
        if x is None and x_shape is None:
            raise ValueError("x and x_shape cannot be both None")
        x_shape = x.shape if x is not None else x_shape
        N, _, H, W = x_shape
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_dino_hat = output["h_dino_hat"].contiguous()
        h_dino = output["h_dino"].contiguous()

        h_dino_loss = F.mse_loss(h_dino_hat[:, 1:, :], h_dino[:, 1:, :])
        cls_token_loss = F.mse_loss(h_dino_hat[:, 0, :], h_dino[:, 0, :])

        rlmbda = self.get_rlmbda(global_step)
        loss = rlmbda * bpp_loss + cls_token_loss + h_dino_loss  # + h_vqgan_loss

        monitor = {
            "loss": loss.detach().mean().item(),
            "cls": cls_token_loss.detach().mean().item(),
            "dino": h_dino_loss.detach().mean().item(),
            "bpp": bpp_loss.item(),
            "rlmbda": rlmbda,
        }
        monitor.update(
            {name: bpp.detach().mean().item() for name, bpp in bpp_components.items()}
        )
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


class MPC12Loss(nn.Module):
    def __init__(
        self,
        rlmbda=24.0,
    ):
        super().__init__()
        self.rlmbda = rlmbda

    def get_rlmbda(self, global_step=None):
        return self.rlmbda

    def forward(self, output, x, x_shape=None, global_step=None):
        if x is None and x_shape is None:
            raise ValueError("x and x_shape cannot be both None")
        x_shape = x.shape if x is not None else x_shape
        N, _, H, W = x_shape
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_vqgan_hat = output["h_vqgan_hat"].contiguous()
        h_vqgan = output["h_vqgan"].contiguous()

        h_vqgan_loss = F.mse_loss(h_vqgan_hat, h_vqgan)

        h_dino_hat = output["h_dino_hat"].contiguous()
        h_dino = output["h_dino"].contiguous()

        h_dino_loss = F.mse_loss(h_dino_hat[:, 1:, :], h_dino[:, 1:, :])
        cls_token_loss = F.mse_loss(h_dino_hat[:, 0, :], h_dino[:, 0, :])

        rlmbda = self.get_rlmbda(global_step)
        loss = rlmbda * bpp_loss + cls_token_loss + h_dino_loss + h_vqgan_loss

        monitor = {
            "loss": loss.detach().mean().item(),
            "cls": cls_token_loss.detach().mean().item(),
            "dino": h_dino_loss.detach().mean().item(),
            "vqgan": h_vqgan_loss.detach().mean().item(),
            "bpp": bpp_loss.item(),
            "rlmbda": rlmbda,
        }
        monitor.update(
            {name: bpp.detach().mean().item() for name, bpp in bpp_components.items()}
        )
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor
