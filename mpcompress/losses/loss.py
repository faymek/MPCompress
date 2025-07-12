import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import Tensor


def kl_loss_for_dist1(
    dist1: DiagonalGaussianDistribution, dist2: DiagonalGaussianDistribution
):
    # special case where only dist1 requires grad
    # logvar is around -15.5+=2.4, too big for torch.exp(-logvar)
    # original kl loss
    # 0.5 * torch.sum(
    #     torch.pow(dist1.mean - dist2.mean, 2) / dist2.var
    #     + dist1.var / dist2.var
    #     - 1.0
    #     - dist1.logvar
    #     + dist2.logvar, dim=[1, 2, 3])

    dist2_logvar = torch.clamp(dist2.logvar + 15.5, min=-3)
    return torch.mean(
        torch.pow(dist1.mean - dist2.mean, 2) * torch.exp(-dist2_logvar)
        + dist1.var * torch.exp(-dist2_logvar)
    )


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


class FitLoss(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def forward(self, output, x):
        h_vqgan_hat = output["h_vqgan_hat"].contiguous()
        h_vqgan = output["h_vqgan"].contiguous()
        loss = F.mse_loss(h_vqgan_hat, h_vqgan)

        monitor = {
            "loss": loss.detach().mean().item(),
        }
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


class VaeDistillLoss(nn.Module):
    def __init__(
        self,
        rlmbda=1.0,
        vae_loss_type="mse",
    ):
        super().__init__()
        self.rlmbda = rlmbda
        self.vae_loss_type = vae_loss_type

    def get_rlmbda(self, global_step=None):
        return self.rlmbda

    def forward(self, output, x, global_step=None):
        N, _, H, W = x.size()
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_vae_dist: DiagonalGaussianDistribution = output["h_vae_dist"]
        if self.vae_loss_type == "mse":
            h_vae_hat: Tensor = output["h_vae_hat"].contiguous()
            h_vae: Tensor = h_vae_dist.sample().clone()
            h_vae_loss = F.mse_loss(h_vae_hat, h_vae)
        elif self.vae_loss_type == "nll":
            h_vae_hat: Tensor = output["h_vae_hat"].contiguous()
            h_vae_mean: Tensor = h_vae_dist.mean.clone()
            h_vae_logvar: Tensor = h_vae_dist.logvar.clone()
            h_vae_logvar = torch.clamp(h_vae_logvar + 15.5, min=-3)
            h_vae_loss = torch.mean(
                torch.pow(h_vae_hat - h_vae_mean, 2) * torch.exp(-h_vae_logvar)
            )
        elif self.vae_loss_type == "kl":
            h_vae_hat_dist: DiagonalGaussianDistribution = output["h_vae_hat_dist"]
            h_vae_loss = kl_loss_for_dist1(h_vae_hat_dist, h_vae_dist)

        else:
            raise ValueError(f"Invalid vae_loss_type: {self.vae_loss_type}")

        rlmbda = self.get_rlmbda(global_step)
        if rlmbda > 0.0:
            loss = rlmbda * bpp_loss + h_vae_loss
        else:
            loss = h_vae_loss

        monitor = {
            "loss": loss.detach().mean().item(),
            "vae": h_vae_loss.detach().mean().item(),
            "bpp": bpp_loss.item(),
            "rlmbda": rlmbda,
        }
        monitor.update(
            {name: bpp.detach().mean().item() for name, bpp in bpp_components.items()}
        )
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


class MPC12VaeLoss(nn.Module):
    def __init__(
        self,
        rlmbda=1.0,
        vae_loss_type="kl",
    ):
        super().__init__()
        self.rlmbda = rlmbda
        self.vae_loss_type = vae_loss_type

    def get_rlmbda(self, global_step=None):
        return self.rlmbda

    def forward(self, output, x, global_step=None):
        N, _, H, W = x.size()
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_vae_dist: DiagonalGaussianDistribution = output["h_vae_dist"]
        if self.vae_loss_type == "mse":
            h_vae_hat: Tensor = output["h_vae_hat"].contiguous()
            h_vae: Tensor = h_vae_dist.sample().clone()
            h_vae_loss = F.mse_loss(h_vae_hat, h_vae)
        elif self.vae_loss_type == "nll":
            h_vae_hat: Tensor = output["h_vae_hat"].contiguous()
            h_vae_mean: Tensor = h_vae_dist.mean.clone()
            h_vae_logvar: Tensor = h_vae_dist.logvar.clone()
            h_vae_logvar = torch.clamp(h_vae_logvar + 15.5, min=-3)
            h_vae_loss = torch.mean(
                torch.pow(h_vae_hat - h_vae_mean, 2) * torch.exp(-h_vae_logvar)
            )
        elif self.vae_loss_type == "kl":
            h_vae_hat_dist: DiagonalGaussianDistribution = output["h_vae_hat_dist"]
            h_vae_loss = kl_loss_for_dist1(h_vae_hat_dist, h_vae_dist)
        else:
            raise ValueError(f"Invalid vae_loss_type: {self.vae_loss_type}")

        h_dino_hat = output["h_dino_hat"].contiguous()
        h_dino = output["h_dino"].contiguous()

        h_dino_loss = F.mse_loss(h_dino_hat[:, 1:, :], h_dino[:, 1:, :])
        cls_token_loss = F.mse_loss(h_dino_hat[:, 0, :], h_dino[:, 0, :])

        rlmbda = self.get_rlmbda(global_step)
        loss = rlmbda * bpp_loss + h_vae_loss + cls_token_loss + h_dino_loss

        monitor = {
            "loss": loss.detach().mean().item(),
            "cls": cls_token_loss.detach().mean().item(),
            "dino": h_dino_loss.detach().mean().item(),
            "vae": h_vae_loss.detach().mean().item(),
            "bpp": bpp_loss.item(),
            "rlmbda": rlmbda,
        }
        monitor.update(
            {name: bpp.detach().mean().item() for name, bpp in bpp_components.items()}
        )
        if "monitor" in output:
            monitor.update(output["monitor"])
        return loss, monitor


class MPC1Loss(nn.Module):
    def __init__(
        self,
        rlmbda=24.0,
    ):
        super().__init__()
        self.rlmbda = rlmbda

    def get_rlmbda(self, global_step=None):
        return self.rlmbda

    def forward(self, output, x, global_step=None):
        N, _, H, W = x.size()
        num_pixels = N * H * W

        bpp_components = {}
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            bpp_components[name] = bpp
        bpp_loss = sum(bpp_components.values())

        h_vqgan_hat = output["h_vqgan_hat"].contiguous()
        h_vqgan = output["h_vqgan"].contiguous()

        h_vqgan_loss = F.mse_loss(h_vqgan_hat, h_vqgan)

        rlmbda = self.get_rlmbda(global_step)
        loss = h_vqgan_loss

        monitor = {
            "loss": loss.detach().mean().item(),
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


class MPC2Loss(nn.Module):
    def __init__(
        self,
        rlmbda=24.0,
    ):
        super().__init__()
        self.rlmbda = rlmbda

    def get_rlmbda(self, global_step=None):
        return self.rlmbda

    def forward(self, output, x, global_step=None):
        N, _, H, W = x.size()
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

    def forward(self, output, x, global_step=None):
        N, _, H, W = x.size()
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
