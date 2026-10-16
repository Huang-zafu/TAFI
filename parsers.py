"""Dataset profiles and command-line configuration for TAFI."""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:
    yaml = None


ROOT = Path(__file__).resolve().parent

SUPPORTED_DATASETS: Tuple[str, ...] = (
    "GOODCBAS",
    "GOODCora",
    "GOODWebKB",
    "ogbn_arxiv",
    "twitch",
    "Elliptic",
)
GOOD_FOLDERS: Tuple[str, ...] = ("GOODCBAS", "GOODCora", "GOODWebKB")


DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "GOODCBAS": {
        "scaler": "standard",
        "predictor_dropout": 0.5,
        "lr_predictor": 0.01,
        "epochs": 200,
        "patience": 50,
        "selection_metric": "accuracy",
        "collapse_guard": True,
        "perturb_budget_fraction": 0.05,
        "inner_steps": 6,
        "num_envs": 3,
        "predictor_num_layers": 3,
        "predictor_hidden_dim": 300,
    },
    "GOODCBAS/concept": {
        "num_envs": 4,
    },
    "GOODCBAS/covariate": {
        "lr_predictor": 0.003,
    },
    "GOODCora": {
        "scaler": "minmax",
        "predictor_dropout": 0.4,
        "lr_predictor": 0.003,
        "epochs": 200,
        "patience": 40,
        "selection_metric": "accuracy",
        "collapse_guard": True,
        "perturb_budget_fraction": 0.05,
        "inner_steps": 6,
        "num_envs": 3,
        "predictor_num_layers": 2,
        "predictor_hidden_dim": 256,
    },
    "GOODCora/concept": {
        "predictor_dropout": 0.6,
        "lr_predictor": 0.003,
        "perturb_budget_fraction": 0.05,
        "predictor_hidden_dim": 512,
        "inner_steps": 10,
    },
    "GOODCora/covariate": {
        "predictor_dropout": 0.4,
        "lr_predictor": 0.002,
        "perturb_budget_fraction": 0.04,
    },
    "GOODCora/degree/concept": {
        "predictor_num_layers": 2,
        "inner_steps": 6,
    },
    "GOODCora/degree/covariate": {
        "num_envs": 4,
        "protection_dropout": 0.7,
    },
    "GOODCora/word/covariate": {
        "predictor_dropout": 0.6,
        "inner_steps": 10,
    },
    "GOODWebKB": {
        "scaler": "minmax",
        "predictor_dropout": 0.7,
        "lr_predictor": 0.05,
        "epochs": 100,
        "patience": 30,
        "selection_metric": "accuracy",
        "collapse_guard": True,
        "perturb_budget_fraction": 0.10,
        "inner_steps": 10,
        "num_envs": 3,
        "predictor_num_layers": 3,
        "predictor_hidden_dim": 300,
    },
    "GOODWebKB/concept": {},
    "GOODWebKB/covariate": {
        "perturb_budget_fraction": 0.04,
        "inner_steps": 5,
        "num_envs": 4,
        "predictor_num_layers": 2,
    },
    "ogbn_arxiv": {
        "scaler": "none",
        "predictor_dropout": 0.1,
        "lr_predictor": 0.01,
        "epochs": 200,
        "patience": 40,
        "selection_metric": "micro_f1",
        "collapse_guard": True,
        "perturb_budget_fraction": 0.08,
        "inner_steps": 8,
        "num_envs": 2,
        "predictor_num_layers": 3,
        "predictor_hidden_dim": 300,
    },
    "twitch": {
        "scaler": "none",
        "predictor_dropout": 0.5,
        "lr_predictor": 0.03,
        "epochs": 200,
        "patience": 50,
        "selection_metric": "roc_auc",
        "collapse_guard": True,
        "perturb_budget_fraction": 0.05,
        "inner_steps": 6,
        "num_envs": 2,
        "predictor_num_layers": 2,
        "predictor_hidden_dim": 256,
    },
    "Elliptic": {
        "scaler": "none",
        "train_mode": "snapshotwise",
        "predictor_hidden_dim": 300,
        "predictor_num_layers": 3,
        "predictor_dropout": 0.5,
        "lr_predictor": 0.01,
        "epochs": 100,
        "patience": 10,
        "selection_metric": "macro_f1",
        "collapse_guard": True,
        "num_envs": 3,
        "perturb_budget_fraction": 0.10,
        "inner_steps": 10,
    },
}


DEFAULT_MODEL_HYPERPARAMS: Dict[str, Any] = {
    "predictor_hidden_dim": 300,
    "predictor_num_layers": 3,
    "num_envs": 3,
    "perturb_budget_fraction": 0.08,
    "protection_hidden_dim": 128,
    "protection_dropout": 0.5,
    "max_edge_attenuation": 0.95,
    "protect_eps": 1e-3,
    "priority_floor": 1e-6,
    "lr_augmenter": 0.002,
    "lr_protection_generator": 0.002,
    "predictor_weight_decay": 1e-4,
    "augmenter_weight_decay": 1e-4,
    "protection_generator_weight_decay": 1e-4,
    "inner_steps": 8,
    "alpha_div": 0.05,
}


TUNABLE_MODEL_KEYS: Tuple[str, ...] = (
    "predictor_hidden_dim",
    "predictor_num_layers",
    "num_envs",
    "perturb_budget_fraction",
    "protection_hidden_dim",
    "protection_dropout",
    "max_edge_attenuation",
    "lr_augmenter",
    "lr_protection_generator",
    "predictor_weight_decay",
    "augmenter_weight_decay",
    "protection_generator_weight_decay",
    "inner_steps",
    "alpha_div",
)


NUMERICAL_MODEL_KEYS: Tuple[str, ...] = ("protect_eps", "priority_floor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TAFI benchmark training and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON/YAML configuration file; CLI options take precedence.",
    )

    data = parser.add_argument_group("Dataset")
    data.add_argument(
        "--datasets",
        default="all",
        help=f"'all' or a comma-separated subset of {SUPPORTED_DATASETS}.",
    )
    data.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Root directory containing benchmark data folders.",
    )
    data.add_argument("--good-shifts", default="concept,covariate")
    data.add_argument("--good-domain", default=None)

    opt = parser.add_argument_group("Optimization")
    opt.add_argument("--epochs", type=int, default=None)
    opt.add_argument("--patience", type=int, default=None)
    opt.add_argument("--lr-predictor", type=float, default=None)
    opt.add_argument("--lr-augmenter", type=float, default=None)
    opt.add_argument("--lr-protection-generator", type=float, default=None)
    opt.add_argument("--predictor-weight-decay", type=float, default=None)
    opt.add_argument("--augmenter-weight-decay", type=float, default=None)
    opt.add_argument("--protection-generator-weight-decay", type=float, default=None)
    opt.add_argument("--inner-steps", type=int, default=None)
    opt.add_argument("--alpha-div", type=float, default=None)
    opt.add_argument(
        "--selection-metric",
        choices=("accuracy", "micro_f1", "macro_f1", "roc_auc"),
        default=None,
        help="Override the validation metric used for checkpoint selection.",
    )
    opt.add_argument(
        "--collapse-guard", choices=("auto", "true", "false"), default="auto"
    )
    opt.add_argument(
        "--scaler", choices=("auto", "standard", "minmax", "none"), default="auto"
    )

    model = parser.add_argument_group("Capacity / environment construction")
    model.add_argument("--predictor-hidden-dim", type=int, default=None)
    model.add_argument("--predictor-num-layers", type=int, default=None)
    model.add_argument("--predictor-dropout", type=float, default=None)
    model.add_argument("--num-envs", type=int, default=None)
    model.add_argument("--perturb-budget-fraction", type=float, default=None)
    model.add_argument("--protection-hidden-dim", type=int, default=None)
    model.add_argument("--protection-dropout", type=float, default=None)
    model.add_argument("--max-edge-attenuation", type=float, default=None)

    numerical = parser.add_argument_group("Numerical stability")
    numerical.add_argument(
        "--protect-eps", type=float, default=None,
        help="Positive margin below full protection (default: 1e-3).",
    )
    numerical.add_argument(
        "--priority-floor", type=float, default=None,
        help="Lower bound for finite-precision editor priorities (default: 1e-6).",
    )

    system = parser.add_argument_group("Execution")
    system.add_argument("--seeds", default="0,1,2,3,4")
    system.add_argument("--device", default="auto")
    system.add_argument(
        "--output-dir", type=Path, default=ROOT / "results_tafi_final"
    )
    system.add_argument("--log-every", type=int, default=10)
    system.add_argument("--no-checkpoint", action="store_true")
    system.add_argument("--save-predictions", action="store_true")
    system.add_argument("--overwrite", action="store_true")
    return parser


def get_profile_for_setting(
    family: str,
    shift: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:

    profile = dict(DEFAULT_PROFILES.get(family, {}))
    keys = []
    if shift:
        keys.append(f"{family}/{shift}")
    if domain:
        keys.append(f"{family}/{domain}")
    if domain and shift:
        keys.append(f"{family}/{domain}/{shift}")
    for key in keys:
        profile.update(DEFAULT_PROFILES.get(key, {}))
    return profile


def load_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configuration files")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    return dict(raw)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(args)
    if pre_args.config is not None:
        parser.set_defaults(**load_config_file(pre_args.config))

    parsed = parser.parse_args(args)
    parsed.data_root = Path(parsed.data_root).expanduser().resolve()
    parsed.output_dir = Path(parsed.output_dir).expanduser().resolve()
    return parsed


def get_hyperparams(args: argparse.Namespace) -> Dict[str, Any]:

    return {
        key: getattr(args, key)
        for key in TUNABLE_MODEL_KEYS + NUMERICAL_MODEL_KEYS
    }


def resolve_hyperparams(
    profile: Mapping[str, Any],
    user_hyperparams: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved = dict(DEFAULT_MODEL_HYPERPARAMS)
    for key in DEFAULT_MODEL_HYPERPARAMS:
        if profile.get(key) is not None:
            resolved[key] = profile[key]
    for key, value in user_hyperparams.items():
        if value is not None:
            resolved[key] = value
    return resolved


def get_profile_overrides(args: argparse.Namespace) -> Dict[str, Any]:

    overrides: Dict[str, Any] = {}
    for key in ("epochs", "patience", "lr_predictor", "predictor_dropout", "selection_metric"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    collapse_guard = args.collapse_guard
    if isinstance(collapse_guard, bool):
        overrides["collapse_guard"] = collapse_guard
    elif collapse_guard != "auto":
        overrides["collapse_guard"] = collapse_guard == "true"

    if args.scaler != "auto":
        overrides["scaler"] = args.scaler
    return overrides


def save_config(args: argparse.Namespace, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)

    suffix = file_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required to save YAML configuration files")
        with file_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(values, handle, sort_keys=False)
    else:
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, ensure_ascii=False)
