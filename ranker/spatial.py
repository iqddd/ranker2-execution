"""Reusable Step40 linear and spatial ranking primitives.

The functions in this module retain Step40's group-major pair order and
mixed-precision boundaries.  Experiment policies and artifact handling stay
in the root-level step scripts.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.nn import functional as F

from ranker.attention_lora import Float32LayerNorm
from ranker.data import RankedImage
from ranker.group_ranknet import ranknet_target
from ranker.heldout_ranknet import IDENTITY_ORDER

def train_prior_pair_classes(
    train_o: Sequence[Sequence[RankedImage]],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return aligned/counter masks in the stable 28-pair order per group."""
    identities = sorted({item.identity for group in train_o for item in group})
    total_count = {identity: 0 for identity in identities}
    total_sum = {identity: 0.0 for identity in identities}
    for group in train_o:
        for item in group:
            total_count[item.identity] += 1
            total_sum[item.identity] += float(item.z)
    aligned_masks: list[np.ndarray] = []
    counter_masks: list[np.ndarray] = []
    for group in train_o:
        local_count = {identity: 0 for identity in identities}
        local_sum = {identity: 0.0 for identity in identities}
        for item in group:
            local_count[item.identity] += 1
            local_sum[item.identity] += float(item.z)
        prior = {
            identity: (total_sum[identity] - local_sum[identity])
            / (total_count[identity] - local_count[identity] + 8.0)
            for identity in identities
        }
        aligned: list[bool] = []
        counter: list[bool] = []
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (
                    (first, second) if first.rank < second.rank else (second, first)
                )
                delta = prior[higher.identity] - prior[lower.identity]
                cross = higher.identity != lower.identity
                aligned.append(bool(cross and delta > 0.0))
                counter.append(bool(cross and delta < 0.0))
        aligned_masks.append(np.asarray(aligned, dtype=bool))
        counter_masks.append(np.asarray(counter, dtype=bool))
    return tuple(aligned_masks), tuple(counter_masks)


def pair_arrays(
    groups: Sequence[Sequence[RankedImage]],
    z: np.ndarray,
    name_to_row: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build feature differences in the original nested group/pair order."""
    differences: list[np.ndarray] = []
    targets: list[float] = []
    distances: list[float] = []
    for group in groups:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lower, higher = (
                    (first, second) if first.rank < second.rank else (second, first)
                )
                distance = higher.rank - lower.rank
                differences.append(
                    z[name_to_row[higher.image_name]] - z[name_to_row[lower.image_name]]
                )
                targets.append(ranknet_target(distance))
                distances.append(float(distance))
    return np.stack(differences), np.asarray(targets), np.asarray(distances)


def linear_pair_coefficients(
    train_o: Sequence[Sequence[RankedImage]],
    masks: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    *,
    balanced: bool,
) -> np.ndarray:
    if not balanced:
        return np.concatenate(
            [
                np.asarray(
                    [
                        float(abs(first.rank - second.rank))
                        for offset, first in enumerate(group)
                        for second in group[offset + 1 :]
                    ],
                    dtype=np.float64,
                )
                / (84.0 * len(train_o))
                for group in train_o
            ]
        )
    aligned_masks, counter_masks = masks
    n_aligned_groups = sum(bool(mask.any()) for mask in aligned_masks)
    n_counter_groups = sum(bool(mask.any()) for mask in counter_masks)
    coefficients: list[np.ndarray] = []
    for aligned, counter in zip(aligned_masks, counter_masks, strict=True):
        values = np.zeros(28, dtype=np.float64)
        if aligned.any():
            values[aligned] = 0.5 / n_aligned_groups / int(aligned.sum())
        if counter.any():
            values[counter] = 0.5 / n_counter_groups / int(counter.sum())
        coefficients.append(values)
    result = np.concatenate(coefficients)
    if not np.isclose(result.sum(), 1.0, atol=1e-12):
        raise AssertionError(
            f"Balanced pair coefficients sum to {result.sum()}, expected 1."
        )
    return result


class CudaLinearObjective:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        coefficient: np.ndarray,
        device: torch.device,
        *,
        l2: float,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float64, device=device)
        self.y = torch.as_tensor(y, dtype=torch.float64, device=device)
        self.coefficient = torch.as_tensor(
            coefficient, dtype=torch.float64, device=device
        )
        self.dimension = int(x.shape[1])
        self.l2 = l2

    def __call__(self, vector: np.ndarray) -> tuple[float, np.ndarray]:
        weight = torch.as_tensor(vector, dtype=torch.float64, device=self.x.device)
        logits = self.x.mv(weight)
        data = torch.sum(
            self.coefficient
            * (torch.logaddexp(torch.zeros_like(logits), logits) - self.y * logits)
        )
        gradient = (
            self.x.t().mv(self.coefficient * (torch.sigmoid(logits) - self.y))
            + 2.0 * self.l2 * weight
        )
        return float(
            (data + self.l2 * torch.dot(weight, weight)).item()
        ), gradient.cpu().numpy()


def fit_cuda_linear(
    objective: CudaLinearObjective,
    *,
    options: Mapping[str, float | int],
    success_gradient_l2: float,
) -> tuple[np.ndarray, dict[str, object]]:
    initial = np.zeros(objective.dimension, dtype=np.float64)
    loss_at_zero = objective(initial)[0]
    result = minimize(
        objective, initial, method="L-BFGS-B", jac=True, options=dict(options)
    )
    weight = np.asarray(result.x, dtype=np.float64)
    final_loss, gradient = objective(weight)
    finite = bool(
        np.all(np.isfinite(weight))
        and np.isfinite(final_loss)
        and np.all(np.isfinite(gradient))
    )
    gradient_l2 = float(np.linalg.norm(gradient))
    return weight, {
        "finite": finite,
        "loss_at_zero": float(loss_at_zero),
        "final_loss": float(final_loss),
        "gradient_l2": gradient_l2,
        "scipy_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_nit": int(result.nit),
        "optimizer_nfev": int(result.nfev),
        "certificate_success": bool(
            finite and final_loss <= loss_at_zero and gradient_l2 <= success_gradient_l2
        ),
    }


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = Float32LayerNorm(channels)
        self.pointwise1 = nn.Linear(channels, 4 * channels)
        self.pointwise2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.pointwise2(F.gelu(self.pointwise1(self.norm(x))))
        x = x * self.gamma.to(dtype=x.dtype)
        return residual + x.permute(0, 3, 1, 2)


class SpatialResidualHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_norm = Float32LayerNorm(1152, elementwise_affine=False)
        self.input_projection = nn.Linear(1152, 128)
        self.stage1 = nn.Sequential(ConvNeXtBlock(128), ConvNeXtBlock(128))
        self.down1_norm = Float32LayerNorm(128)
        self.down1 = nn.Conv2d(128, 192, 2, stride=2)
        self.stage2 = ConvNeXtBlock(192)
        self.down2_norm = Float32LayerNorm(192)
        self.down2 = nn.Conv2d(192, 256, 2, stride=2)
        self.stage3 = ConvNeXtBlock(256)
        self.head_norm = Float32LayerNorm(512)
        self.head1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.20)
        self.output = nn.Linear(256, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _downsample(
        x: torch.Tensor, norm: Float32LayerNorm, convolution: nn.Conv2d
    ) -> torch.Tensor:
        return convolution(norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

    def forward(
        self, tokens: torch.Tensor, *, diagnostics: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        x = self.input_norm(tokens)
        x = self.input_projection(x).reshape(-1, 32, 32, 128).permute(0, 3, 1, 2)
        x = self.stage1(x)
        x = self._downsample(x, self.down1_norm, self.down1)
        x = self.stage2(x)
        x = self._downsample(x, self.down2_norm, self.down2)
        feature_map = self.stage3(x)
        feature_map_fp32 = feature_map.float()
        with torch.autocast(device_type="cuda", enabled=False):
            mean_pool = feature_map_fp32.mean(dim=(2, 3))
            max_pool = feature_map_fp32.amax(dim=(2, 3))
        pooled = torch.cat((mean_pool, max_pool), dim=1)
        hidden = self.dropout(F.gelu(self.head1(self.head_norm(pooled))))
        delta = self.output(hidden).squeeze(1).float()
        raw = (
            {
                "feature_map": feature_map_fp32,
                "mean_pool": mean_pool,
                "max_pool": max_pool,
            }
            if diagnostics
            else None
        )
        return delta, raw


@dataclass(frozen=True)
class SpatialFoldData:
    fold_index: int
    heldout: tuple[str, str]
    train_o: tuple[tuple[RankedImage, ...], ...]
    train_names: tuple[str, ...]
    global_indices: torch.Tensor
    base_scores: torch.Tensor
    higher: torch.Tensor
    lower: torch.Tensor
    targets: torch.Tensor
    coefficients: torch.Tensor
    n_validation_counter_pairs: int


def build_spatial_fold_data(
    *,
    fold_index: int,
    heldout: tuple[str, str],
    source_groups: Sequence[Sequence[RankedImage]],
    masks: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    name_to_global: Mapping[str, int],
    linear_bal_scores: np.ndarray,
    observations: Mapping[str, np.ndarray],
    device: torch.device,
) -> SpatialFoldData:
    heldout_set = set(heldout)
    train_o = tuple(
        sorted(
            (
                tuple(group)
                for group in source_groups
                if not any(item.identity in heldout_set for item in group)
            ),
            key=lambda group: group[0].group_index,
        )
    )
    train_names = tuple(item.image_name for group in train_o for item in group)
    if len(train_names) != len(set(train_names)):
        raise ValueError("Step 40 requires disjoint train-O images.")
    name_to_local = {name: index for index, name in enumerate(train_names)}
    higher: list[int] = []
    lower: list[int] = []
    targets: list[float] = []
    for group in train_o:
        for offset, first in enumerate(group):
            for second in group[offset + 1 :]:
                lo, hi = (
                    (first, second) if first.rank < second.rank else (second, first)
                )
                higher.append(name_to_local[hi.image_name])
                lower.append(name_to_local[lo.image_name])
                targets.append(ranknet_target(hi.rank - lo.rank))
    coefficients = linear_pair_coefficients(train_o, masks, balanced=True).astype(
        np.float32
    )
    pair_indices = tuple(sorted(IDENTITY_ORDER.index(name) for name in heldout))
    validation_mask = (
        (observations["identity_1"] == pair_indices[0])
        & (observations["identity_2"] == pair_indices[1])
        & (observations["delta_prior"] < 0.0)
    )
    global_indices_np = np.asarray(
        [name_to_global[name] for name in train_names], dtype=np.int64
    )
    return SpatialFoldData(
        fold_index=fold_index,
        heldout=heldout,
        train_o=train_o,
        train_names=train_names,
        global_indices=torch.as_tensor(
            global_indices_np, dtype=torch.long, device=device
        ),
        base_scores=torch.as_tensor(
            linear_bal_scores[fold_index, global_indices_np],
            dtype=torch.float32,
            device=device,
        ),
        higher=torch.as_tensor(higher, dtype=torch.long, device=device),
        lower=torch.as_tensor(lower, dtype=torch.long, device=device),
        targets=torch.as_tensor(targets, dtype=torch.float32, device=device),
        coefficients=torch.as_tensor(coefficients, dtype=torch.float32, device=device),
        n_validation_counter_pairs=int(np.sum(validation_mask)),
    )


def spatial_balanced_loss(scores: torch.Tensor, fold: SpatialFoldData) -> torch.Tensor:
    with torch.autocast(device_type="cuda", enabled=False):
        values = scores.float()
        logits = values.index_select(0, fold.higher) - values.index_select(
            0, fold.lower
        )
        losses = F.binary_cross_entropy_with_logits(
            logits, fold.targets, reduction="none"
        )
        return torch.sum(fold.coefficients * losses)


def spatial_forward_scores(
    model: SpatialResidualHead,
    tokens: torch.Tensor,
    fold: SpatialFoldData,
    *,
    mixed_precision: bool,
    groupwise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks: list[torch.Tensor] = []
    ranges = range(0, len(fold.global_indices), 8) if groupwise else (0,)
    for start in ranges:
        indices = (
            fold.global_indices[start : start + 8] if groupwise else fold.global_indices
        )
        selected = tokens.index_select(0, indices)
        if not mixed_precision:
            selected = selected.float()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision
        ):
            current, _ = model(selected)
        chunks.append(current)
    delta = torch.cat(chunks)
    with torch.autocast(device_type="cuda", enabled=False):
        delta_fp32 = delta.float()
        scores = fold.base_scores.float() + delta_fp32
    return scores, delta_fp32


def initialize_model(seed: int, device: torch.device) -> SpatialResidualHead:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SpatialResidualHead().to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("All spatial master parameters must be float32.")
    if (
        int(torch.count_nonzero(model.output.weight)) != 0
        or int(torch.count_nonzero(model.output.bias)) != 0
    ):
        raise AssertionError(
            "Production spatial scalar layer must be zero initialized."
        )
    return model


def flattened_gradients(model: nn.Module) -> torch.Tensor:
    """Flatten gradients in parameter order, replacing missing gradients with FP32 zeros."""
    values: list[torch.Tensor] = []
    for parameter in model.parameters():
        values.append(
            torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
            if parameter.grad is None
            else parameter.grad.detach().float().reshape(-1)
        )
    return torch.cat(values)


def cpu_float32_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return finite FP32 checkpoint tensors without changing state-dict key order."""
    state = {
        name: value.detach().to(device="cpu", dtype=torch.float32)
        for name, value in model.state_dict().items()
    }
    if any(
        value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("Checkpoint parameters must be finite float32 tensors.")
    return state


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return correlation with the existing zero-variance fallback."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if len(x) < 2 or x.std(ddof=0) == 0.0 or y.std(ddof=0) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def certify_mixed_precision(
    tokens: torch.Tensor, fold: SpatialFoldData, *, device: torch.device
) -> dict[str, object]:
    """Run Step40's fixed FP32/BF16 parity certification for one spatial fold."""
    seed = 20260722 + 1000 * fold.fold_index + 1
    initialized = initialize_model(seed, device).eval()
    generator = torch.Generator(device="cpu").manual_seed(20260722)
    temporary_weight = (
        torch.randn(initialized.output.weight.shape, generator=generator) * 1e-3
    )
    with torch.no_grad():
        initialized.output.weight.copy_(temporary_weight.to(device))
        initialized.output.bias.zero_()
    fp32_model = copy.deepcopy(initialized)
    bf16_model = copy.deepcopy(initialized)
    del initialized

    def evaluate(
        model: SpatialResidualHead, mixed: bool
    ) -> tuple[float, np.ndarray, np.ndarray, int]:
        model.eval()
        model.zero_grad(set_to_none=True)
        scores, _ = spatial_forward_scores(
            model, tokens, fold, mixed_precision=mixed, groupwise=True
        )
        loss = spatial_balanced_loss(scores, fold)
        loss.backward()
        gradients = flattened_gradients(model)
        nonfinite = int((~torch.isfinite(scores)).sum().item())
        nonfinite += int(not bool(torch.isfinite(loss)))
        nonfinite += int((~torch.isfinite(gradients)).sum().item())
        return (
            float(loss.detach().cpu()),
            scores.detach().cpu().numpy().astype(np.float64),
            gradients.cpu().numpy().astype(np.float64),
            nonfinite,
        )

    loss_fp32, score_fp32, gradient_fp32, nonfinite_fp32 = evaluate(fp32_model, False)
    del fp32_model
    torch.cuda.empty_cache()
    loss_bf16, score_bf16, gradient_bf16, nonfinite_bf16 = evaluate(bf16_model, True)
    del bf16_model
    torch.cuda.empty_cache()
    absolute_loss_difference = abs(loss_bf16 - loss_fp32)
    relative_loss_difference = absolute_loss_difference / max(abs(loss_fp32), 1e-12)
    gradient_denominator = max(float(np.linalg.norm(gradient_fp32)), 1e-12)
    gradient_cosine = float(
        np.dot(gradient_fp32, gradient_bf16)
        / max(np.linalg.norm(gradient_fp32) * np.linalg.norm(gradient_bf16), 1e-12)
    )
    report: dict[str, object] = {
        "fold_index": fold.fold_index,
        "heldout_pair": list(fold.heldout),
        "train_O_groups": len(fold.train_o),
        "train_images": len(fold.train_names),
        "run_index": 1,
        "loss_FP32": loss_fp32,
        "loss_BF16": loss_bf16,
        "absolute_loss_difference": absolute_loss_difference,
        "relative_loss_difference": relative_loss_difference,
        "score_spearman": float(spearmanr(score_fp32, score_bf16).statistic),
        "score_pearson": float(pearsonr(score_fp32, score_bf16).statistic),
        "gradient_cosine": gradient_cosine,
        "relative_gradient_L2_difference": float(
            np.linalg.norm(gradient_bf16 - gradient_fp32) / gradient_denominator
        ),
        "n_nonfinite_FP32": nonfinite_fp32,
        "n_nonfinite_BF16": nonfinite_bf16,
    }
    report["mixed_precision_certification_pass"] = bool(
        nonfinite_fp32 == 0
        and nonfinite_bf16 == 0
        and relative_loss_difference <= 1e-3
        and report["score_spearman"] >= 0.999
        and report["score_pearson"] >= 0.999
        and gradient_cosine >= 0.995
        and report["relative_gradient_L2_difference"] <= 0.05
    )
    return report
