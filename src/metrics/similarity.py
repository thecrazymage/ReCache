"""Reference-based image metrics."""

from types import SimpleNamespace

import torch
import lpips
import torch.nn.functional as F
from einops import rearrange
from munch import Munch

from src.metrics.base import BaseMetric


class LPIPSMetric(BaseMetric):
    """LPIPS distance to the references; lower is better.

    Adapted from https://github.com/richzhang/PerceptualSimilarity.

    Args:
        net: LPIPS backbone, `'vgg'`, `'alex'` or `'squeeze'`.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        net: str = 'vgg',
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)

        self.model = lpips.LPIPS(net=net, spatial=False).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if references is None:
            raise ValueError(f'{self} requires references.')

        # [0, 1] -> [-1, 1]
        gen = (generated.to(self.device).clamp(0, 1) * 2.0) - 1.0
        ref = (references.to(self.device).clamp(0, 1) * 2.0) - 1.0

        scores = self.model(ref, gen).squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        return scores

    def __str__(self) -> str:
        return 'LPIPS'


class PLPIPSMetric(BaseMetric):
    """Patch-wise LPIPS distance to the references; lower is better.

    Images are split into overlapping square patches, padded by edge
    replication so that the patches cover the whole image, and LPIPS is
    averaged over patches.

    Args:
        patch_size: Patch side in pixels.
        stride: Step between patches in pixels.
        net: LPIPS backbone, `'vgg'`, `'alex'` or `'squeeze'`.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        patch_size: int = 224,
        stride: int = 160,
        net: str = 'vgg',
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)

        self.model = lpips.LPIPS(net=net, spatial=False).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.patch_size = patch_size
        self.stride = stride

    def _extract(self, images: torch.Tensor) -> torch.Tensor:
        """Split images into patches of shape (B * num_patches, C, P, P)."""
        pad_h = (self.patch_size - images.shape[-2]) % self.stride
        pad_w = (self.patch_size - images.shape[-1]) % self.stride
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode='replicate')

        patches = images.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        # patches: [B, C, NumPatchesH, NumPatchesW, PatchH, PatchW]
        return rearrange(patches, 'b c nh nw h w -> (b nh nw) c h w')

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if references is None:
            raise ValueError(f'{self} requires references.')

        B = generated.shape[0]

        # [0, 1] -> [-1, 1]
        gen = (generated.to(self.device).clamp(0, 1) * 2.0) - 1.0
        ref = (references.to(self.device).clamp(0, 1) * 2.0) - 1.0

        # Extract patches
        gen_patches = self._extract(gen)
        ref_patches = self._extract(ref)

        # Compute LPIPS on patches
        scores = self.model(ref_patches, gen_patches).view(
            B, -1
        )  # [B, num_patches]
        scores = scores.mean(dim=1)

        return scores

    def __str__(self) -> str:
        return 'PLPIPS'


class SSIMMetric(BaseMetric):
    """SSIM to the references; higher is better.

    Gaussian-window SSIM with zero-padded borders, averaged over pixels
    and channels. The window shrinks for images smaller than it.

    Args:
        data_range: Value range of the images.
        window_size: Side of the Gaussian window.
        window_sigma: Standard deviation of the Gaussian window.
        k1: SSIM constant of the luminance term.
        k2: SSIM constant of the contrast-structure term.
        eps: Lower bound on the SSIM denominator.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        data_range: float = 1.0,
        window_size: int = 11,
        window_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)

        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.window_sigma = float(window_sigma)
        self.k1 = float(k1)
        self.k2 = float(k2)

        self._window = None
        self._window_size = None
        self._window_channel = None
        self._window_device = None
        self._window_dtype = None

        self.eps = float(eps)

    def _create_window(
        self,
        window_size: int,
        channel: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a normalized Gaussian window of shape (C, 1, K, K)."""
        coords = (
            torch.arange(window_size, device=device, dtype=dtype)
            - window_size // 2
        )
        g = torch.exp(-(coords**2) / (2 * (self.window_sigma**2)))
        g = g / g.sum()
        window_2d = g[:, None] * g[None, :]
        window_2d = window_2d / window_2d.sum()
        window = window_2d.expand(
            channel, 1, window_size, window_size
        ).contiguous()
        return window

    def _get_window(
        self,
        window_size: int,
        channel: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the cached window, rebuilding it if its parameters
        change."""
        if (
            self._window is None
            or self._window_size != window_size
            or self._window_channel != channel
            or self._window_device != device
            or self._window_dtype != dtype
        ):
            self._window = self._create_window(
                window_size, channel, device, dtype
            )
            self._window_size = window_size
            self._window_channel = channel
            self._window_device = device
            self._window_dtype = dtype
        return self._window

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if references is None:
            raise ValueError(f'{self} requires references.')

        gen = generated.to(self.device).clamp(0, 1)
        ref = references.to(self.device).clamp(0, 1)

        _, c, h, w = gen.shape
        window_size = min(self.window_size, h, w)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 1:
            raise ValueError(
                "SSIM window_size is too small for the input images."
            )

        window = self._get_window(window_size, c, gen.device, gen.dtype)
        padding = window_size // 2

        mu1 = F.conv2d(gen, window, padding=padding, groups=c)
        mu2 = F.conv2d(ref, window, padding=padding, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(gen * gen, window, padding=padding, groups=c) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(ref * ref, window, padding=padding, groups=c) - mu2_sq
        )
        sigma12 = (
            F.conv2d(gen * ref, window, padding=padding, groups=c) - mu1_mu2
        )

        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2

        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        denominator = torch.clamp(denominator, min=self.eps)
        ssim_map = numerator / denominator

        scores = ssim_map.mean(dim=(1, 2, 3))

        if self.is_train:
            scores = -scores

        return scores

    def __str__(self) -> str:
        return 'SSIM'


class PSNRMetric(BaseMetric):
    """PSNR to the references in dB; higher is better.

    Args:
        data_range: Value range of the images.
        eps: Lower bound on the MSE, so that PSNR stays finite.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        data_range: float = 1.0,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)

        self.data_range = float(data_range)
        self.eps = float(eps)

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if references is None:
            raise ValueError(f'{self} requires references.')

        gen = generated.to(self.device).clamp(0, 1)
        ref = references.to(self.device).clamp(0, 1)

        mse = (gen - ref).pow(2).flatten(1).mean(dim=1)
        mse = torch.clamp(mse, min=self.eps)
        data_range = torch.as_tensor(
            self.data_range,
            device=mse.device,
            dtype=mse.dtype,
        )
        scores = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse)

        if self.is_train:
            scores = -scores

        return scores

    def __str__(self) -> str:
        return 'PSNR'


class L1Metric(BaseMetric):
    """Mean absolute error to the references; lower is better."""

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if references is None:
            raise ValueError(f'{self} requires references.')

        diff = torch.abs(generated.clamp(0, 1) - references.clamp(0, 1))
        scores = diff.view(diff.size(0), -1).mean(dim=1)
        return scores

    def __str__(self) -> str:
        return 'L1'
