"""Reusable latent selection utilities for tilt-style agents."""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


ScoreFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
SampleZFn = Callable[[int], torch.Tensor]


@dataclass
class TiltLatentSelector:
    """Maintains and refreshes a latent pool using a task-coverage score."""

    z: torch.Tensor
    beta: float = 0.995
    temperature: float = 20.0
    candidate_multiplier: int = 10

    def __post_init__(self) -> None:
        dim = self.z.shape[-1]
        self.gram = torch.eye(dim, device=self.z.device, dtype=self.z.dtype)
        self.running_mean = torch.zeros(dim, device=self.z.device, dtype=self.z.dtype)

    @torch.no_grad()
    def refresh(
        self,
        init_features: torch.Tensor,
        sample_z: SampleZFn,
        score_fn: ScoreFn,
    ) -> torch.Tensor:
        n = self.z.shape[0]
        n_candidates = self.candidate_multiplier * n
        z_candidates = sample_z(n_candidates)

        obs_idx = torch.randint(
            0, init_features.shape[0], (n_candidates,), device=init_features.device
        )
        feature_candidates = init_features[obs_idx]

        candidate_score, feature_stats = score_fn(feature_candidates, z_candidates)
        logits = candidate_score / self.temperature
        logits = logits - logits.max()
        prob = torch.softmax(logits, dim=0)
        selected_idx = torch.multinomial(prob, num_samples=n, replacement=False)

        gram_batch = feature_stats[selected_idx].T @ feature_stats[selected_idx] / n
        self.gram.mul_(self.beta).add_((1 - self.beta) * gram_batch)
        self.z = z_candidates[selected_idx]
        return self.z
