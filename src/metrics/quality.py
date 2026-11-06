"""Reference-free image metrics."""

from types import SimpleNamespace

import clip
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from munch import Munch
from hpsv2.img_score import model_dict
from hpsv2.src.open_clip import get_tokenizer, create_model_and_transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    Normalize,
    InterpolationMode,
)

from src.metrics.base import BaseMetric


class AQMetric(BaseMetric):
    """LAION aesthetic score of images; higher is better.

    A linear head on normalized CLIP image features. Adapted from
    https://github.com/LAION-AI/aesthetic-predictor.

    Args:
        model_name_or_path: Path to the linear head weights, downloaded
            by `scripts/download.sh`.
        clip_model_name: CLIP backbone name.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        model_name_or_path: str = (
            'pretrained/aesthetic_model/sa_0_4_vit_l_14_linear.pth'
        ),
        clip_model_name: str = 'ViT-L/14',
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)

        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.model = torch.nn.Linear(768, 1).to(self.device, self.dtype)
        self.model.load_state_dict(
            torch.load(model_name_or_path, map_location=self.device)
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.image_transform = self._clip_transform(224)

    def _clip_transform(self, n_px: int) -> Compose:
        """Resize, center-crop and normalize images for CLIP."""
        return Compose(
            [
                Resize(
                    n_px,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=False,
                ),
                CenterCrop(n_px),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        image_batch = self.image_transform(
            generated.to(self.device).clamp(0, 1)
        )

        image_feats = self.clip_model.encode_image(image_batch).to(self.dtype)
        image_feats = F.normalize(image_feats, dim=-1, p=2)

        scores = self.model(image_feats).squeeze(dim=-1)
        if self.is_train:
            scores = -scores

        return scores

    def __str__(self) -> str:
        return 'AQ'


class CLIPMetric(BaseMetric):
    """CLIP score between images and prompts; higher is better.

    The score is the CLIP image-text logit, i.e. the cosine similarity
    scaled by the CLIP logit scale (100). Adapted from
    https://github.com/openai/CLIP.

    Args:
        model_name: CLIP model name.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        model_name: str = "ViT-L/14",
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)

        self.model, self.preprocess = clip.load(
            model_name, device=self.device, jit=False
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if prompts is None:
            raise ValueError(f'{self} requires prompts.')

        processed_gen = torch.stack(
            [
                self.preprocess(to_pil_image(gen.detach().cpu().clamp(0, 1)))
                for gen in generated
            ]
        ).to(self.device)

        text_tokens = clip.tokenize(prompts, truncate=True).to(self.device)

        logits_per_image, _ = self.model(processed_gen, text_tokens)
        scores = logits_per_image.diagonal()
        if self.is_train:
            scores = -scores

        return scores

    def __str__(self) -> str:
        return 'CLIP'


class HPSMetric(BaseMetric):
    """HPSv2.1 human preference score for prompts; higher is better.

    The model is cached in the global `hpsv2.img_score.model_dict` and
    shared by all instances; `dtype` is ignored in favor of the model
    dtype. Adapted from https://github.com/tgxs002/HPSv2.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        super().__init__(config, is_train, is_video, dtype, device)

        self.hps_precision = "amp"
        self.hps_init_on_cpu = True

        if not model_dict:
            model, preprocess = self._load_model()
            model_dict.update(model=model, preprocess_val=preprocess)

        self.model = model_dict['model'].to(self.device).eval()
        self.preprocess = model_dict['preprocess_val']
        for param in self.model.parameters():
            param.requires_grad = False
        self.dtype = next(self.model.parameters()).dtype
        self.tokenizer = get_tokenizer("ViT-H-14")

    def _create_model(
        self, precision: str, init_device: torch.device | str
    ) -> tuple[torch.nn.Module, Compose]:
        """Create the HPSv2.1 model and load its checkpoint."""
        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=precision,
            device=init_device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        ckpt = hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
        state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)

        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False

        return model, preprocess_val

    def _load_model(self) -> tuple[torch.nn.Module, Compose]:
        init_device = "cpu" if self.hps_init_on_cpu else self.device
        return self._create_model(self.hps_precision, init_device)

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if prompts is None:
            raise ValueError(f'{self} requires prompts.')

        images = torch.stack(
            [
                self.preprocess(to_pil_image(img.detach().cpu().clamp(0, 1)))
                for img in generated
            ]
        ).to(self.device, dtype=self.dtype)

        texts = self.tokenizer(prompts).to(self.device)

        feats = self.model(images, texts)
        scores = (
            feats["image_features"] @ feats["text_features"].T
        ).diagonal()
        if self.is_train:
            scores = -scores

        return scores

    def __str__(self) -> str:
        return 'HPS'


class IRMetric(BaseMetric):
    """ImageReward score for prompts; higher is better.

    Images are scored one by one. Adapted from
    https://github.com/zai-org/ImageReward.

    Args:
        model_name_or_path: ImageReward checkpoint name or path.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        model_name_or_path: str = "ImageReward-v1.0",
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        import ImageReward as RM

        super().__init__(config, is_train, is_video, dtype, device)

        self.model = RM.load(
            model_name_or_path,
            device=device,
            download_root='pretrained/ImageReward',
        ).eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        if prompts is None:
            raise ValueError(f'{self} requires prompts.')

        scores = []
        for img, prompt in zip(generated, prompts):
            score = self.model.score(
                prompt, to_pil_image(img.detach().cpu().clamp(0, 1))
            )
            scores.append(score)

        scores = torch.tensor(scores, device=self.device)
        if self.is_train:
            scores = -scores

        return scores

    def __str__(self) -> str:
        return 'IR'


class IQMetric(BaseMetric):
    """MUSIQ image quality score, as in VBench; higher is better.

    Adapted from https://github.com/Vchitect/VBench.

    Args:
        model_name_or_path: Path to the MUSIQ checkpoint trained on SPAQ.

        See `BaseMetric` for the other arguments.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        model_name_or_path: str = (
            'pretrained/image_quality/musiq_spaq_ckpt-358bb6af.pth'
        ),
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        from pyiqa.archs.musiq_arch import MUSIQ

        super().__init__(config, is_train, is_video, dtype, device)

        self.model = MUSIQ(pretrained_model_path=model_name_or_path).to(
            self.device
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def iq_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Downscale images so that the longer side is at most 512."""
        _, _, h, w = images.size()
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            images = Resize(
                size=(int(h * scale), int(w * scale)), antialias=False
            )(images)
        return images

    @torch.inference_mode()
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        images = self.iq_transform(generated.clamp(0, 1)).to(self.device)

        scores = self.model(images).reshape(-1)
        if self.is_train:
            scores = -scores

        return scores

    def __str__(self) -> str:
        return 'IQ'
