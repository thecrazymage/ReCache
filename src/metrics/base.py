"""Base class for image and video metrics."""

from abc import ABC, abstractmethod
from types import SimpleNamespace

import torch
from munch import Munch
from tqdm import tqdm


class BaseMetric(ABC):
    """Base class for evaluation metrics and training objectives.

    Subclasses implement `_calculate_batch`, which returns one score per
    sample. With `is_train=True`, metrics where higher is better return
    negated scores, so that lower is always better.

    Args:
        config: Metric config; only used by combined metrics.
        is_train: Whether the metric is used as a training objective.
        is_video: Whether the metric scores videos; 5D inputs are then
            scored frame by frame outside training.
        dtype: Dtype of the metric model, where applicable.
        device: Device to run the metric on.
    """

    def __init__(
        self,
        config: Munch | SimpleNamespace | None = None,
        is_train: bool = False,
        is_video: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device('cuda'),
    ) -> None:
        self.config = config
        self.is_train = is_train
        self.is_video = is_video

        self.device = device
        self.dtype = dtype

    def _apply_reduction(
        self, scores: torch.Tensor, reduction: str = 'mean'
    ) -> torch.Tensor:
        if reduction == 'mean':
            return scores.mean()

        if reduction == 'sum':
            return scores.sum()

        if reduction == 'none':
            return scores

        raise NotImplementedError(
            f'unknown reduction method {reduction} from metric '
            f'{self.__class__.__name__}'
        )

    def _validate_video_inputs(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> tuple[int, int]:
        """Check video input shapes and return `(batch_size, num_frames)`.

        Raises:
            ValueError: If the inputs are not 5D, their batch and frame
                dimensions differ, or the number of prompts is neither
                B nor B * T.
        """

        if not isinstance(generated, torch.Tensor) or generated.ndim != 5:
            raise ValueError(
                "When is_video=True, generated videos must have shape "
                f"[B, T, C, H, W], but got {generated.shape}."
            )

        batch_size, num_frames = generated.shape[:2]

        if references is not None:
            if (
                not isinstance(references, torch.Tensor)
                or references.ndim != 5
            ):
                raise ValueError(
                    "When is_video=True, reference videos must have shape "
                    f"[B, T, C, H, W], but got {references.shape}."
                )
            if references.shape[:2] != (batch_size, num_frames):
                raise ValueError(
                    "generated and reference videos must have the same "
                    f"[B, T] dimensions, but got {generated.shape} "
                    f"vs {references.shape}."
                )

        if prompts is not None and not isinstance(prompts, str):
            if (
                len(prompts) != batch_size
                and len(prompts) != batch_size * num_frames
            ):
                raise ValueError(
                    "When is_video=True, prompts must have length B or B*T, "
                    f"but got {len(prompts)}."
                )

        return batch_size, num_frames

    @staticmethod
    def _get_video_prompt(
        prompts: str | list[str] | None,
        batch_idx: int,
        frame_idx: int,
        batch_size: int,
        num_frames: int,
    ) -> list[str] | None:
        """Pick the prompt of one frame from shared, per-video or
        per-frame prompts."""
        if prompts is None:
            return None
        if isinstance(prompts, str):
            return [prompts]
        if len(prompts) == batch_size:
            return [prompts[batch_idx]]
        return [prompts[batch_idx * num_frames + frame_idx]]

    def _calculate_video_frame_by_frame(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        """Score each frame as a separate image and average over frames.

        Args:
            generated: Videos of shape (B, T, C, H, W) in [0, 1].
            references: Reference videos of the same shape, or None.
            prompts: A shared prompt, one prompt per video or one per
                frame, or None.

        Returns:
            Scores of shape (B,).
        """

        batch_size, num_frames = self._validate_video_inputs(
            generated, references, prompts
        )

        video_scores = []
        prev_is_video = self.is_video

        try:
            self.is_video = False

            for batch_idx in range(batch_size):
                frame_scores = []

                for frame_idx in range(num_frames):
                    frame_gen = generated[batch_idx:batch_idx + 1, frame_idx]
                    frame_teach = (
                        references[batch_idx:batch_idx + 1, frame_idx]
                        if references is not None
                        else None
                    )
                    frame_prompts = self._get_video_prompt(
                        prompts, batch_idx, frame_idx, batch_size, num_frames
                    )

                    score = self._calculate_batch(
                        generated=frame_gen,
                        references=frame_teach,
                        prompts=frame_prompts,
                    )

                    if isinstance(score, torch.Tensor):
                        score = score.reshape(-1)
                        if score.numel() != 1:
                            raise ValueError(
                                "Per-frame metric evaluation must return "
                                "exactly one score."
                            )
                        frame_score = score[0]
                    else:
                        frame_score = torch.tensor(
                            float(score),
                            device=self.device,
                            dtype=self.dtype,
                        )

                    frame_scores.append(frame_score)

                video_scores.append(torch.stack(frame_scores).mean())

        finally:
            self.is_video = prev_is_video

        scores = torch.stack(video_scores)
        return scores

    @abstractmethod
    def _calculate_batch(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
    ) -> torch.Tensor:
        """Compute per-sample scores for a batch of images.

        Args:
            generated: Generated images of shape (B, C, H, W) in [0, 1].
            references: Reference images of the same shape, for
                reference-based metrics.
            prompts: Prompts, for text-conditioned metrics.

        Returns:
            Scores of shape (B,).

        Raises:
            ValueError: If the metric needs `references` or `prompts`
                and they are None.
        """
        raise NotImplementedError()

    def calculate_dataset(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
        batch_size: int = 1,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the metric over a whole dataset in batches.

        Args:
            generated: Generated images of shape (N, C, H, W) or videos of
                shape (N, T, C, H, W), in [0, 1].
            references: References of the same shape, for reference-based
                metrics.
            prompts: One prompt per sample or a single shared prompt, for
                text-conditioned metrics.
            batch_size: Number of samples per metric call.
            reduction: `'mean'` or `'sum'` over samples.

        Returns:
            Scalar metric value.
        """
        if references is not None:
            assert len(generated) == len(
                references
            ), 'references size should be the same as generated size'
        if prompts is not None:
            assert len(generated) == len(
                prompts
            ), 'prompts size should be the same as generated size'

        use_video_frame_by_frame = (
            self.is_video
            and not self.is_train
            and isinstance(generated, torch.Tensor)
            and generated.ndim == 5
        )
        calculate_batch = (
            self._calculate_video_frame_by_frame
            if use_video_frame_by_frame
            else self._calculate_batch
        )

        num_generated = len(generated)

        total_score = torch.tensor(0.0, device=self.device)
        num_samples = 0

        for i in tqdm(
            range(0, num_generated, batch_size),
            leave=False,
            desc=f'Computing {self.__class__.__name__}',
        ):
            batch_gen = generated[i:i + batch_size].to(self.device)
            batch_ref = (
                references[i:i + batch_size].to(self.device)
                if references is not None
                else None
            )

            if prompts is None:
                batch_prompts = None
            elif isinstance(prompts, str):
                batch_prompts = prompts
            else:
                batch_prompts = prompts[i:i + batch_size]

            scores = calculate_batch(
                generated=batch_gen,
                references=batch_ref,
                prompts=batch_prompts,
            )

            total_score += scores.sum()
            num_samples += scores.numel()

        del generated
        del references
        torch.cuda.empty_cache()

        if reduction == 'mean':
            return total_score / num_samples
        if reduction == 'sum':
            return total_score

        raise NotImplementedError(
            'unknown reduction type passed for dataset calculation '
            f'{reduction}'
        )

    def __call__(
        self,
        generated: torch.Tensor,
        references: torch.Tensor | None = None,
        prompts: str | list[str] | None = None,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the metric for a single batch.

        See `calculate_dataset` for the arguments; `reduction` also
        accepts `'none'`, which returns per-sample scores of shape (B,).
        """
        if (
            self.is_video
            and not self.is_train
            and isinstance(generated, torch.Tensor)
            and generated.ndim == 5
        ):
            scores = self._calculate_video_frame_by_frame(
                generated=generated, references=references, prompts=prompts
            )
        else:
            scores = self._calculate_batch(
                generated=generated, references=references, prompts=prompts
            )

        return self._apply_reduction(scores, reduction)
