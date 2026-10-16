"""Targeted Adversarial Feature Filtering for Invariance (TAFI)."""


from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


_HC_BETA = 1.0 / 3.0
_HC_GAMMA = -0.2
_HC_ZETA = 1.2
_NUMERICAL_EPS = 1e-12


def _masked_ce(logits: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    if int(mask.sum()) == 0:
        raise ValueError("mask selects no nodes")
    return F.cross_entropy(logits[mask], y[mask])


def _add_remaining_self_loops(
    edge_index: Tensor,
    edge_weight: Tensor,
    num_nodes: int,
    fill_value: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    src, dst = edge_index
    loop_mask = src == dst
    existing_loop_nodes = src[loop_mask]
    has_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    if existing_loop_nodes.numel() > 0:
        has_loop[existing_loop_nodes] = True
    missing = (~has_loop).nonzero(as_tuple=False).flatten()
    if missing.numel() == 0:
        return edge_index, edge_weight
    loop_edges = torch.stack([missing, missing], dim=0)
    loop_weight = edge_weight.new_full((missing.numel(),), float(fill_value))
    return (
        torch.cat([edge_index, loop_edges], dim=1),
        torch.cat([edge_weight, loop_weight], dim=0),
    )


def _nonself_mask(edge_index: Tensor) -> Tensor:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    return edge_index[0] != edge_index[1]


@dataclass
class PredictorOutput:
    logits: Tensor
    feature_gate: Tensor


@dataclass
class _PredictorMultiViewOutput:
    logits: Tensor
    feature_gate: Tensor


@dataclass
class MultiEnvironmentOutput:
    logits: Tensor
    feature_gate: Tensor


def environment_risks(logits_by_env: Tensor, y: Tensor, train_mask: Tensor) -> Tensor:
    return torch.stack(
        [_masked_ce(logits_by_env[m], y, train_mask) for m in range(logits_by_env.size(0))]
    )


def invariance_loss(risks: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

    if risks.dim() != 1 or risks.numel() == 0:
        raise ValueError("risks must be a nonempty one-dimensional tensor")
    l_cls = risks.mean()
    risk_variance = (risks - l_cls).square().mean()
    return l_cls + risk_variance, l_cls, risk_variance


def inner_objective(
    logits_by_env: Tensor,
    y: Tensor,
    train_mask: Tensor,
    alpha_div: float = 0.05,
) -> Tuple[Tensor, Dict[str, Tensor]]:


    if not 0.0 <= float(alpha_div) < float("inf"):
        raise ValueError("alpha_div must be finite and nonnegative")
    risks = environment_risks(logits_by_env, y, train_mask)
    l_cls = risks.mean()
    risk_variance = (risks - l_cls).square().mean()
    l_div = -risk_variance
    l_inner = l_cls + float(alpha_div) * l_div
    return l_inner, {
        "loss_inner": l_inner,
        "loss_cls": l_cls,
        "loss_div": l_div,
        "environment_risk_variance": risk_variance,
        "environment_risks": risks,
    }


def _capped_proportional_fixed_mass(
    raw_priority: Tensor,
    total_mass: float | Tensor,
    *,
    upper: float = 0.95,
) -> Tensor:
    """Allocate capped fixed-mass priorities using a differentiable active set."""


    if raw_priority.dim() != 1:
        raise ValueError("raw_priority must be one-dimensional")
    if not raw_priority.is_floating_point():
        raise ValueError("raw_priority must have a floating-point dtype")
    if not math.isfinite(float(upper)) or not 0.0 < float(upper) <= 1.0:
        raise ValueError("upper must lie in (0,1]")
    if not bool(torch.isfinite(raw_priority).all()) or bool((raw_priority < 0).any()):
        raise ValueError("raw_priority must be finite and nonnegative")
    output_dtype = (
        torch.float64 if raw_priority.dtype == torch.float64 else torch.float32
    )
    raw = raw_priority.to(torch.float64)
    target = (
        total_mass.to(device=raw.device, dtype=torch.float64)
        if isinstance(total_mass, Tensor)
        else raw.new_tensor(float(total_mass))
    )
    if target.numel() != 1 or not bool(torch.isfinite(target).all()):
        raise ValueError("total_mass must be a finite scalar")
    target = target.reshape(())
    if bool(target < 0):
        raise ValueError("total_mass must be nonnegative")
    if bool(target == 0):
        return (raw * 0.0).to(output_dtype)

    positive = raw > 0
    n_positive = int(positive.sum())
    capacity = float(upper) * n_positive
    if bool(target > capacity):
        raise ValueError(
            "fixed-mass allocation is infeasible on the positive-priority support: "
            f"target={float(target.detach()):.17g}, capacity={capacity:.17g}"
        )

    if bool(target == capacity):
        delta = torch.where(positive, raw * 0.0 + float(upper), raw * 0.0)
    else:
        with torch.no_grad():
            indices = positive.nonzero(as_tuple=False).flatten()
            order = torch.argsort(raw[indices], descending=True)
            sorted_indices = indices[order]
            sorted_priority = raw[sorted_indices]
            log_priority = sorted_priority.log()
            log_suffix_sum = torch.logcumsumexp(log_priority.flip(0), dim=0).flip(0)
            n_saturated = torch.arange(n_positive, device=raw.device, dtype=raw.dtype)
            remaining_by_set = target.detach() - float(upper) * n_saturated
            largest_candidate = remaining_by_set * (log_priority - log_suffix_sum).exp()
            feasible_sets = (remaining_by_set > 0) & (largest_candidate <= float(upper))
            feasible_indices = feasible_sets.nonzero(as_tuple=False).flatten()
            if feasible_indices.numel() == 0:
                raise RuntimeError("no numerically valid active set for fixed-mass allocation")
            count = int(feasible_indices[0])
            saturated = torch.zeros_like(positive)
            saturated[sorted_indices[:count]] = True
            active = positive & ~saturated

        active_priority = raw[active]
        # Common scaling preserves the allocation exactly. Its detached value is
        # a numerical unit choice; the denominator below still carries gradients.
        remaining = target - float(upper) * count
        if active_priority.numel() == 1:
            # The last unsaturated edge receives the remaining mass independently
            # of its priority, including when that priority is subnormal.
            active_delta = active_priority * 0.0 + remaining
        else:
            normalized = active_priority / active_priority.detach().max()
            active_delta = remaining * (normalized / normalized.sum())
        if bool((active_delta.detach() > float(upper) * (1.0 + 1e-12)).any()):
            raise RuntimeError("active-set allocation exceeds the per-edge cap")
        delta = raw * 0.0
        delta = delta.masked_fill(saturated, float(upper))
        delta = delta.masked_scatter(active, active_delta.clamp_max(float(upper)))

    result = delta.to(output_dtype)
    mass_error = (result.detach().sum(dtype=torch.float64) - target.detach()).abs()
    tolerance = max(
        torch.finfo(torch.float64).tiny,
        16.0 * torch.finfo(output_dtype).eps * abs(float(target.detach())),
    )
    if (
        not bool(torch.isfinite(result).all())
        or bool((result < 0).any())
        or bool((result > result.new_tensor(float(upper))).any())
        or float(mass_error) > tolerance
    ):
        raise RuntimeError(
            "fixed-mass allocation failed numerical validation: "
            f"budget_error={float(mass_error):.6g}, tolerance={tolerance:.6g}"
        )
    return result


def _fixed_mass_retention(delta: Tensor, total_mass: float | Tensor, *, upper: float) -> Tensor:
    """Return retention weights while checking numerical budget preservation."""


    retention = 1.0 - delta.to(torch.float64)
    represented_delta = 1.0 - retention
    target = torch.as_tensor(total_mass, device=delta.device, dtype=torch.float64)
    dtype_eps = torch.finfo(delta.dtype).eps
    target_value = abs(float(target.detach()))
    # Subtraction near one has an absolute, rather than delta-relative, rounding
    # error. Bound its accumulation, while still requiring small relative error.
    subtraction_tolerance = min(
        4.0 * torch.finfo(torch.float64).eps * delta.size(-1),
        1e-5 * target_value,
    )
    tolerance = max(
        torch.finfo(torch.float64).tiny,
        16.0 * dtype_eps * target_value,
        subtraction_tolerance,
    )
    mass_error = (represented_delta.detach().sum(dim=-1) - target.detach()).abs()
    if (
        not bool(torch.isfinite(retention).all())
        or bool((represented_delta < 0).any())
        or bool((represented_delta > float(upper) * (1.0 + 16.0 * dtype_eps)).any())
        or bool(((delta > 0) & (represented_delta == 0)).any())
        or bool((mass_error > tolerance).any())
    ):
        raise RuntimeError(
            "fixed-mass retention failed numerical validation; "
            "the attenuation or budget is not representable accurately"
        )
    return retention


class ComplementAdversarialGraphEditor(nn.Module):


    def __init__(
        self,
        num_envs: int,
        num_edges: int,
        *,
        perturb_budget_fraction: float = 0.2,
        max_edge_attenuation: float = 0.95,
        priority_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_envs < 2:
            raise ValueError("num_envs must be >= 2")
        if num_edges < 0:
            raise ValueError("num_edges must be nonnegative")
        if not 0.0 <= float(perturb_budget_fraction) < float(max_edge_attenuation):
            raise ValueError(
                "perturb_budget_fraction must be nonnegative and smaller than max_edge_attenuation"
            )
        if not 0.0 < float(max_edge_attenuation) <= 1.0:
            raise ValueError("max_edge_attenuation must lie in (0,1]")
        if not 0.0 < float(priority_floor) < 1.0:
            raise ValueError("priority_floor must lie in (0,1)")
        if float(torch.tensor(float(priority_floor), dtype=torch.float32)) == 0.0:
            raise ValueError("priority_floor must be representable as positive float32")
        self.num_envs = int(num_envs)
        self.num_edges = int(num_edges)
        self.perturb_budget_fraction = float(perturb_budget_fraction)
        self.max_edge_attenuation = float(max_edge_attenuation)
        self.priority_floor = float(priority_floor)
        self.B = nn.Parameter(torch.empty(self.num_envs, self.num_edges))
        nn.init.uniform_(self.B)

    def _target_mass(self, edge_index: Tensor) -> float:
        return self.perturb_budget_fraction * int(_nonself_mask(edge_index).sum())

    def _delta_for_view(self, k: int, edge_index: Tensor) -> Tensor:
        if edge_index.size(1) != self.num_edges:
            raise ValueError("graph edge count does not match augmenter construction")
        nonself = _nonself_mask(edge_index)
        if int(nonself.sum()) == 0 or self.perturb_budget_fraction == 0.0:
            return self.B[int(k)].to(
                torch.float64 if self.B.dtype == torch.float64 else torch.float32
            ) * 0.0
        logits = self.B[int(k)]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("editor logits must be finite")
        logits = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
        # Finite sigmoid logits can underflow to zero in floating-point arithmetic.
        # The floor preserves support; its clamped region has zero logit gradient.
        proposal = torch.sigmoid(logits).clamp_min(self.priority_floor)
        raw = torch.where(nonself, proposal, torch.zeros_like(proposal))
        target_mass = self._target_mass(edge_index)
        delta = _capped_proportional_fixed_mass(
            raw, target_mass, upper=self.max_edge_attenuation
        )
        if bool((delta[nonself] <= 0).any()):
            raise RuntimeError("positive adversarial allocation underflowed; increase precision or budget")
        return delta

    def edge_attenuation(self, edge_index: Tensor) -> Tensor:
        return torch.stack(
            [self._delta_for_view(k, edge_index) for k in range(self.num_envs)], dim=0
        )


class SharedH0ProtectionGenerator(nn.Module):


    def __init__(
        self,
        h0_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        dropout: float = 0.5,
        protect_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 < float(protect_eps) < 1.0:
            raise ValueError("protect_eps must lie in (0,1)")
        if float(torch.tensor(1.0 - float(protect_eps), dtype=torch.float32)) == 1.0:
            raise ValueError("protect_eps must make 1-protect_eps strictly below one in float32")
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * int(h0_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(num_layers)),
        )
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.protect_eps = float(protect_eps)

    def logits(self, h_init_detached: Tensor, edge_index: Tensor) -> Tensor:
        z = F.dropout(h_init_detached, p=self.dropout, training=self.training)
        src, dst = edge_index
        return self.edge_mlp(torch.cat([z[src], z[dst]], dim=-1))

    @staticmethod
    def _relative_center(logits: Tensor, edge_index: Tensor) -> Tensor:
        logits = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
        nonself = _nonself_mask(edge_index)
        if int(nonself.sum()) == 0:
            return logits
        return logits - logits[nonself].mean(dim=0, keepdim=True)

    def _hard_concrete(
        self, logits: Tensor, edge_index: Tensor, *, stochastic: Optional[bool]
    ) -> Tensor:
        if stochastic is None:
            stochastic = self.training
        logits = logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("protection logits must be finite")
        if stochastic:
            u = torch.empty_like(logits).uniform_(1e-6, 1.0 - 1e-6)
            binary = torch.sigmoid(
                (torch.log(u) - torch.log1p(-u) + logits) / _HC_BETA
            )
        else:
            binary = torch.sigmoid(logits)
        stretched = binary * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
        clipped = stretched.clamp(0.0, 1.0)
        gate = (
            stretched + (clipped - stretched).detach()
            if self.training and stochastic
            else clipped
        ).transpose(0, 1)
        gate = (1.0 - self.protect_eps) * gate
        nonself = _nonself_mask(edge_index).unsqueeze(0)
        return torch.where(nonself, gate, torch.ones_like(gate))

    def forward(
        self,
        h_init_detached: Tensor,
        edge_index: Tensor,
        *,
        stochastic: Optional[bool] = None,
    ) -> Tensor:
        if not bool(_nonself_mask(edge_index).any()):

            dtype = torch.float64 if h_init_detached.dtype == torch.float64 else torch.float32
            return torch.ones(
                (self.num_layers, edge_index.size(1)),
                device=h_init_detached.device,
                dtype=dtype,
            )
        effective = self._relative_center(self.logits(h_init_detached, edge_index), edge_index)
        return self._hard_concrete(effective, edge_index, stochastic=stochastic)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.projector = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projector(F.dropout(x, p=self.dropout, training=self.training))


class AFFGate(nn.Module):


    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.node_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
        )
        self.channel_gate = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, h_init: Tensor) -> Tensor:
        return torch.sigmoid(self.channel_gate(self.node_net(h_init)))


class PreAggregationGCNConv(nn.Module):


    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        gate: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        num_edges = edge_index.size(1)
        if gate.shape != x.shape:
            raise ValueError("gate must have the same shape as x")
        if edge_weight.shape != (num_edges,):
            raise ValueError("edge weights must have shape [E]")

        full_index, full_weight = _add_remaining_self_loops(
            edge_index, edge_weight, x.size(0), fill_value=1.0
        )
        src, dst = full_index
        current = full_weight.to(x.dtype).clamp(0.0, 1.0)
        # Weight arithmetic stays in the predictor dtype. Reject a material loss
        # of attenuation mass during this cast rather than silently using a
        # different budget (especially when tiny attenuations round to zero).
        requested_mass = (1.0 - full_weight.detach().to(torch.float64)).sum()
        represented_mass = (1.0 - current.detach().to(torch.float64)).sum()
        cast_tolerance = max(
            torch.finfo(torch.float64).tiny,
            16.0 * torch.finfo(x.dtype).eps * abs(float(requested_mass)),
        )
        if (
            not bool(torch.isfinite(requested_mass))
            or not bool(torch.isfinite(represented_mass))
            or float((requested_mass - represented_mass).abs()) > cast_tolerance
        ):
            raise RuntimeError(
                "edge-weight cast cannot preserve the attenuation budget in the predictor dtype: "
                f"requested={float(requested_mass):.6g}, represented={float(represented_mass):.6g}, "
                f"tolerance={cast_tolerance:.6g}; use higher precision or a representable budget"
            )
        degree = current.new_zeros(x.size(0))
        degree.index_add_(0, dst, current)
        inv_sqrt = degree.clamp_min(_NUMERICAL_EPS).pow(-0.5)
        norm = current * inv_sqrt[src] * inv_sqrt[dst]

        edge_gate = gate[src]
        self_loop = src == dst
        if bool(self_loop.any()):
            edge_gate = torch.where(
                self_loop.view(-1, 1), torch.ones_like(edge_gate), edge_gate
            )
        messages = norm.view(-1, 1) * (edge_gate * x[src])
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, dst, messages)
        return self.linear(aggregated)


class TAFIPredictor(nn.Module):


    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.encoder = FeatureEncoder(in_dim, hidden_dim, dropout)
        self.feature_gate = AFFGate(hidden_dim)
        self.convs = nn.ModuleList(
            PreAggregationGCNConv(hidden_dim) for _ in range(self.num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)
        )
        self.depth_logits = nn.Parameter(torch.randn(self.num_layers))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _dropout_mask(x: Tensor, p: float, training: bool) -> Optional[Tensor]:
        if not training or p <= 0.0:
            return None
        keep = 1.0 - p
        return x.new_empty(x.shape).bernoulli_(keep).div_(keep)

    @staticmethod
    def _apply_mask(x: Tensor, mask: Optional[Tensor]) -> Tensor:
        return x if mask is None else x * mask

    def encode_and_gate(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h_init = self.encoder(x)
        return h_init, self.feature_gate(h_init)

    def forward_views_layerwise(
        self,
        x: Tensor,
        edge_index: Tensor,
        layer_edge_weight_views: Sequence[Sequence[Optional[Tensor]]],
    ) -> _PredictorMultiViewOutput:
        if not layer_edge_weight_views:
            raise ValueError("at least one graph view is required")
        if any(len(view) != self.num_layers for view in layer_edge_weight_views):
            raise ValueError("one edge-weight tensor is required for every GNN layer")

        h_init, gate = self.encode_and_gate(x)
        reference = torch.ones(
            edge_index.size(1), device=edge_index.device, dtype=h_init.dtype
        )
        masks = tuple(
            self._dropout_mask(h_init, p=self.dropout, training=self.training)
            for _ in range(self.num_layers)
        )
        num_views = len(layer_edge_weight_views)
        hs = [h_init for _ in range(num_views)]
        layer_states: List[List[Tensor]] = []

        for layer in range(self.num_layers):
            updated: List[Tensor] = []
            for view_idx, edge_weights in enumerate(layer_edge_weight_views):
                current_weight = edge_weights[layer]
                if current_weight is None:
                    current_weight = reference
                next_h = self.convs[layer](
                    hs[view_idx],
                    edge_index,
                    gate=gate,
                    edge_weight=current_weight,
                )
                next_h = F.relu(self.norms[layer](next_h))
                next_h = self._apply_mask(next_h, masks[layer])
                updated.append(next_h)
            hs = updated
            layer_states.append(list(hs))

        depth_alpha = F.softmax(self.depth_logits, dim=0)
        finals: List[Tensor] = []
        for view_idx in range(num_views):
            stacked = torch.stack(
                [layer_states[d][view_idx] for d in range(self.num_layers)], dim=0
            )
            h_depth = torch.sum(stacked * depth_alpha.view(-1, 1, 1), dim=0)
            final_h = h_depth + h_init
            finals.append(final_h)

        logits = torch.stack([self.classifier(h) for h in finals], dim=0)
        return _PredictorMultiViewOutput(logits=logits, feature_gate=gate)

    def forward_views(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight_views: Sequence[Optional[Tensor]],
    ) -> _PredictorMultiViewOutput:
        layer_views = [
            [edge_weight for _ in range(self.num_layers)] for edge_weight in edge_weight_views
        ]
        return self.forward_views_layerwise(x, edge_index, layer_views)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
    ) -> PredictorOutput:
        out = self.forward_views(x, edge_index, [edge_weight])
        return PredictorOutput(logits=out.logits[0], feature_gate=out.feature_gate)


class TAFI(nn.Module):


    def __init__(
        self,
        *,
        in_dim: int,
        predictor_hidden_dim: int,
        num_classes: int,
        num_edges: int,
        predictor_num_layers: int,
        predictor_dropout: float,
        num_envs: int = 3,
        protection_hidden_dim: int = 128,
        protection_dropout: float = 0.5,
        perturb_budget_fraction: float = 0.08,
        max_edge_attenuation: float = 0.95,
        protect_eps: float = 1e-3,
        priority_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(num_envs) < 2:
            raise ValueError("num_envs must be >= 2")
        if not 0.0 < float(perturb_budget_fraction):
            raise ValueError("perturb_budget_fraction must be positive")
        if not 0.0 < float(max_edge_attenuation) <= 1.0:
            raise ValueError("max_edge_attenuation must lie in (0,1]")

        self.num_envs = int(num_envs)
        self.num_layers = int(predictor_num_layers)
        self.perturb_budget_fraction = float(perturb_budget_fraction)
        self.max_edge_attenuation = float(max_edge_attenuation)
        self.protect_eps = float(protect_eps)
        self.priority_floor = float(priority_floor)

        self.predictor = TAFIPredictor(
            in_dim=in_dim,
            hidden_dim=predictor_hidden_dim,
            num_classes=num_classes,
            num_layers=predictor_num_layers,
            dropout=predictor_dropout,
        )
        self.augmenter = ComplementAdversarialGraphEditor(
            num_envs=num_envs,
            num_edges=num_edges,
            perturb_budget_fraction=perturb_budget_fraction,
            max_edge_attenuation=max_edge_attenuation,
            priority_floor=priority_floor,
        )
        self.protection_generator = SharedH0ProtectionGenerator(
            h0_dim=predictor_hidden_dim,
            hidden_dim=protection_hidden_dim,
            num_layers=predictor_num_layers,
            dropout=protection_dropout,
            protect_eps=protect_eps,
        )


    def predictor_parameters(self) -> Iterable[nn.Parameter]:
        return self.predictor.parameters()

    def augmenter_parameters(self) -> Iterable[nn.Parameter]:
        return self.augmenter.parameters()

    def protection_generator_parameters(self) -> Iterable[nn.Parameter]:
        return self.protection_generator.parameters()

    def set_inner_loop(self) -> None:
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        for p in self.protection_generator.parameters():
            p.requires_grad_(False)
        for p in self.augmenter.parameters():
            p.requires_grad_(True)

    def set_outer_loop(self) -> None:
        for p in self.augmenter.parameters():
            p.requires_grad_(False)
        for p in self.predictor.parameters():
            p.requires_grad_(True)
        for p in self.protection_generator.parameters():
            p.requires_grad_(True)

    def forward_source_output(self, x: Tensor, edge_index: Tensor) -> PredictorOutput:
        return self.predictor(x, edge_index, edge_weight=None)

    def forward_source(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward_source_output(x, edge_index).logits

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward_source(x, edge_index)

    def forward_adversarial_only(
        self, x: Tensor, edge_index: Tensor
    ) -> MultiEnvironmentOutput:
        delta = self.augmenter.edge_attenuation(edge_index)
        edge_views = list(
            _fixed_mass_retention(delta, self._target_mass(edge_index), upper=self.max_edge_attenuation)
        )
        out = self.predictor.forward_views(x, edge_index, edge_views)
        return MultiEnvironmentOutput(logits=out.logits, feature_gate=out.feature_gate)

    def structural_protection_scores(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        stochastic: Optional[bool] = None,
    ) -> Tensor:

        with torch.no_grad():
            h_init_detached = self.predictor.encoder(x)
        return self.protection_generator(
            h_init_detached,
            edge_index,
            stochastic=stochastic,
        )

    def _target_mass(self, edge_index: Tensor) -> Tensor:
        return torch.tensor(
            self.perturb_budget_fraction * int(_nonself_mask(edge_index).sum()),
            device=self.augmenter.B.device,
            dtype=torch.float64,
        )

    def _priority_is_feasible(
        self, priority: Tensor, nonself: Tensor, target_mass: Tensor
    ) -> bool:

        if not bool(torch.isfinite(priority).all()) or bool((priority < 0).any()):
            return False
        masked = torch.where(nonself, priority, torch.zeros_like(priority))
        positive_count = int((masked > 0).sum())
        capacity = float(self.max_edge_attenuation * positive_count)
        target = float(target_mass.detach())
        return math.isfinite(target) and 0.0 <= target <= capacity

    def _allocate_fixed_mass(
        self,
        *,
        targeted_priority: Tensor,
        adversarial_priority: Tensor,
        edge_index: Tensor,
        target_mass: Tensor,
    ) -> Tensor:
        """Only targeted_priority is used; adversarial_priority is an unused compatibility argument."""


        nonself = _nonself_mask(edge_index)
        targeted = torch.where(
            nonself, targeted_priority, torch.zeros_like(targeted_priority)
        )
        delta = _capped_proportional_fixed_mass(
            targeted, target_mass, upper=self.max_edge_attenuation
        )
        return torch.where(nonself, delta, torch.zeros_like(delta))

    def _targeted_fixed_mass_retention(
        self,
        *,
        layer_protection: Tensor,
        adv_delta: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        target = self._target_mass(edge_index)
        if not bool(torch.isfinite(layer_protection).all()) or bool(
            ((layer_protection < 0) | (layer_protection > 1)).any()
        ):
            raise ValueError("layer_protection must be finite and lie in [0,1]")
        if not bool(torch.isfinite(adv_delta).all()) or bool((adv_delta < 0).any()):
            raise ValueError("adv_delta must be finite and nonnegative")
        raw = (1.0 - layer_protection.unsqueeze(0)) * adv_delta
        nonself = _nonself_mask(edge_index)
        if bool(target > 0) and bool((raw[:, nonself] <= 0).any()):
            raise RuntimeError("target priorities lost positive support; check protection scores and precision")
        deltas = [
            self._allocate_fixed_mass(
                targeted_priority=raw[k],
                adversarial_priority=adv_delta[k],
                edge_index=edge_index,
                target_mass=target,
            )
            for k in range(self.num_envs)
        ]
        return _fixed_mass_retention(
            torch.stack(deltas, dim=0), target, upper=self.max_edge_attenuation
        )

    def forward_targeted_environments(
        self, x: Tensor, edge_index: Tensor
    ) -> MultiEnvironmentOutput:
        layer_protection = self.structural_protection_scores(
            x, edge_index, stochastic=None
        )
        adv_delta = self.augmenter.edge_attenuation(edge_index).detach()
        retention_layers = [
            self._targeted_fixed_mass_retention(
                layer_protection=layer_protection[layer],
                adv_delta=adv_delta,
                edge_index=edge_index,
            )
            for layer in range(self.num_layers)
        ]

        layer_views = [
            [retention_layers[layer][view] for layer in range(self.num_layers)]
            for view in range(self.num_envs)
        ]
        out = self.predictor.forward_views_layerwise(x, edge_index, layer_views)
        return MultiEnvironmentOutput(logits=out.logits, feature_gate=out.feature_gate)

    def outer_objective(
        self, x: Tensor, edge_index: Tensor, labels: Tensor, train_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        augmented = self.forward_targeted_environments(x, edge_index)
        risks = environment_risks(augmented.logits, labels, train_mask)
        loss_outer, loss_cls, risk_variance = invariance_loss(risks)
        return loss_outer, {
            "loss_outer": loss_outer,
            "loss_cls": loss_cls,
            "environment_risk_variance": risk_variance,
            "environment_risks": risks,
        }


__all__ = ["TAFI", "inner_objective", "invariance_loss"]
