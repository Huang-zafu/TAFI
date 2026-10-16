"""Training and evaluation runner for TAFI benchmarks."""


from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import model_tafi as tafi
from dataset import load_nc_dataset

from parsers import (
    DEFAULT_MODEL_HYPERPARAMS,
    GOOD_FOLDERS,
    SUPPORTED_DATASETS,
    get_hyperparams,
    get_profile_overrides,
    parse_args,
    get_profile_for_setting,
    resolve_hyperparams,
    save_config,
)


@dataclass
class GraphBundle:
    x: Any
    edge_index: Any
    y: Any
    train_mask: Any
    val_mask: Any
    test_mask: Any
    name: str
    source_path: str = ""


@dataclass
class ExperimentSetting:
    key: str
    family: str
    domain: str
    shift: str
    train: GraphBundle
    val: GraphBundle
    tests: List[Tuple[str, GraphBundle]]
    profile: Dict[str, Any]
    main_metric: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def sha_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: scalarize(r.get(k, "")) for k in fieldnames})
    tmp.replace(path)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return float(variance ** 0.5)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_graph(bundle: GraphBundle, device: Any) -> GraphBundle:
    return GraphBundle(
        x=bundle.x.to(device),
        edge_index=bundle.edge_index.to(device),
        y=bundle.y.to(device),
        train_mask=bundle.train_mask.to(device),
        val_mask=bundle.val_mask.to(device),
        test_mask=bundle.test_mask.to(device),
        name=bundle.name,
        source_path=bundle.source_path,
    )


def clone_state(module: Any) -> Dict[str, Any]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            cleaned = {}
            for k, v in r.items():
                if v == "" or v is None:
                    cleaned[k] = None
                elif v.lower() == "true":
                    cleaned[k] = True
                elif v.lower() == "false":
                    cleaned[k] = False
                else:
                    try:
                        cleaned[k] = int(v) if v.isdigit() or (v.startswith('-') and v[1:].isdigit()) else float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append(cleaned)
        return rows


def merge_run_records(existing: List[Dict[str, Any]], new_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    record_map: Dict[Any, Dict[str, Any]] = {}
    for r in existing:
        key = (str(r.get("setting", "")), str(r.get("seed", "")))
        record_map[key] = r
    for r in new_runs:
        key = (str(r.get("setting", "")), str(r.get("seed", "")))
        record_map[key] = r
    return sorted(record_map.values(), key=lambda x: (str(x.get("setting", "")), int(x.get("seed", 0))))


def scalarize(value: Any) -> Any:
    if torch is not None and isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): scalarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalarize(v) for v in value]
    return value


def _extract_pyg_data(path: Path) -> Any:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        data, _slices = loaded
    else:
        data = loaded
    required = ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask")
    missing = [name for name in required if getattr(data, name, None) is None]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")
    return data


def _fit_transform_good_features(x: Any, train_mask: Any, scaler_name: str) -> Any:
    x = x.detach().cpu().float()
    if scaler_name == "none":
        return x
    raw = x.numpy()
    train_np = train_mask.detach().cpu().numpy().astype(bool)
    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    scaler.fit(raw[train_np])
    return torch.as_tensor(scaler.transform(raw), dtype=torch.float32)


def load_good_setting(
    folder: str,
    domain: str,
    shift: str,
    path: Path,
    profile_override: Dict[str, Any],
) -> ExperimentSetting:
    data = _extract_pyg_data(path)
    y = data.y.squeeze(-1).long() if data.y.dim() > 1 else data.y.long()
    train_mask = data.train_mask.squeeze().bool()
    val_mask = data.val_mask.squeeze().bool()
    test_mask = data.test_mask.squeeze().bool()

    p = get_profile_for_setting(folder, shift, domain)
    p.update(profile_override)

    x = _fit_transform_good_features(data.x, train_mask, str(p["scaler"]))
    graph = GraphBundle(
        x=x,
        edge_index=data.edge_index.long().cpu(),
        y=y.cpu(),
        train_mask=train_mask.cpu(),
        val_mask=val_mask.cpu(),
        test_mask=test_mask.cpu(),
        name=f"{folder}/{domain}/{shift}",
        source_path=str(path),
    )
    return ExperimentSetting(
        key=f"{folder}/{domain}/{shift}",
        family=folder,
        domain=domain,
        shift=shift,
        train=graph,
        val=graph,
        tests=[("OOD_test", graph)],
        profile=p,
        main_metric=str(p["selection_metric"]),
    )


def discover_good_settings(
    data_root: Path,
    target_folders: Sequence[str],
    shifts: Sequence[str],
    domain_filter: Optional[str] = None,
) -> List[Tuple[str, str, str, Path]]:
    found: List[Tuple[str, str, str, Path]] = []
    for folder in target_folders:
        if folder not in GOOD_FOLDERS:
            continue
        root = data_root / folder
        if not root.is_dir():
            print(f"Notice: GOOD folder not found at {root}, skipping.")
            continue
        for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if domain_filter and domain_filter.lower() not in domain_dir.name.lower():
                continue
            processed = domain_dir / "processed"
            if not processed.is_dir():
                continue
            for shift in shifts:
                path = processed / f"{shift}.pt"
                if path.is_file():
                    found.append((folder, domain_dir.name, shift, path))
    return found


def nc_to_graph(dataset: Any, name: str, target_mask: Optional[Any] = None) -> GraphBundle:
    graph = dataset.graph
    x = torch.as_tensor(graph["node_feat"], dtype=torch.float32).cpu()
    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long).cpu()
    y = torch.as_tensor(dataset.label).squeeze(-1).long().cpu()
    n = int(graph["num_nodes"])

    if target_mask is None:
        candidate = getattr(dataset, "test_mask", None)
        if candidate is None:
            target_mask = torch.ones(n, dtype=torch.bool)
        else:
            target_mask = torch.as_tensor(candidate).squeeze().bool().cpu()
    else:
        target_mask = torch.as_tensor(target_mask).squeeze().bool().cpu()

    if target_mask.numel() != n:
        raise ValueError(f"{name}: target mask length {target_mask.numel()} != num_nodes {n}")

    return GraphBundle(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=target_mask.clone(),
        val_mask=target_mask.clone(),
        test_mask=target_mask.clone(),
        name=name,
        source_path="dataset.load_nc_dataset",
    )


def load_ogbn_arxiv_setting(data_root: Path, profile_override: Dict[str, Any]) -> Optional[ExperimentSetting]:
    folder = data_root / "ogbn_arxiv"
    if not folder.is_dir():
        print(f"Notice: ogbn_arxiv folder not found at {folder}, skipping.")
        return None

    tr_year = [1950, 2011]
    va_year = [2011, 2014]
    te_years = ([2014, 2016], [2016, 2018], [2018, 2020])

    tr_ds = load_nc_dataset(data_dir=folder, dataname="ogb-arxiv", year=tr_year)
    va_ds = load_nc_dataset(data_dir=folder, dataname="ogb-arxiv", year=va_year)
    te_ds = [load_nc_dataset(data_dir=folder, dataname="ogb-arxiv", year=list(y)) for y in te_years]

    train = nc_to_graph(tr_ds, "ogbn_arxiv/train_1950_2011")
    val = nc_to_graph(va_ds, "ogbn_arxiv/val_2011_2014")
    tests = [(f"{a}_{b}", nc_to_graph(ds, f"ogbn_arxiv/test_{a}_{b}")) for ds, (a, b) in zip(te_ds, te_years)]

    p = get_profile_for_setting("ogbn_arxiv", "temporal")
    p.update(profile_override)

    return ExperimentSetting(
        key="ogbn_arxiv/year/temporal",
        family="ogbn_arxiv",
        domain="year",
        shift="temporal",
        train=train,
        val=val,
        tests=tests,
        profile=p,
        main_metric=str(p["selection_metric"]),
    )


def load_twitch_setting(data_root: Path, profile_override: Dict[str, Any]) -> Optional[ExperimentSetting]:
    folder = data_root / "twitch"
    if not folder.is_dir():
        print(f"Notice: twitch folder not found at {folder}, skipping.")
        return None

    train_name = "DE"
    val_name = "ENGB"
    test_names = ("ES", "FR", "PTBR", "RU", "TW")

    tr_ds = load_nc_dataset(data_dir=folder, dataname="twitch-e", sub_dataname=train_name)
    va_ds = load_nc_dataset(data_dir=folder, dataname="twitch-e", sub_dataname=val_name)
    te_ds = [load_nc_dataset(data_dir=folder, dataname="twitch-e", sub_dataname=name) for name in test_names]

    train = nc_to_graph(tr_ds, "twitch/train_DE")
    val = nc_to_graph(va_ds, "twitch/val_ENGB")
    tests = [(name, nc_to_graph(ds, f"twitch/test_{name}")) for name, ds in zip(test_names, te_ds)]

    p = get_profile_for_setting("twitch", "OOD_countries")
    p.update(profile_override)

    return ExperimentSetting(
        key="twitch/DE_ENGB/OOD_countries",
        family="twitch",
        domain="DE_ENGB",
        shift="OOD_countries",
        train=train,
        val=val,
        tests=tests,
        profile=p,
        main_metric=str(p["selection_metric"]),
    )


def load_selected_settings(
    data_root: Path,
    selected_datasets: Sequence[str],
    good_shifts: Sequence[str],
    good_domain: Optional[str] = None,
    profile_override: Optional[Dict[str, Any]] = None,
) -> List[ExperimentSetting]:
    if profile_override is None:
        profile_override = {}

    settings: List[ExperimentSetting] = []
    

    good_targets = [d for d in selected_datasets if d in GOOD_FOLDERS]
    if good_targets:
        good_discovered = discover_good_settings(data_root, good_targets, good_shifts, good_domain)
        for folder, domain, shift, path in good_discovered:
            settings.append(load_good_setting(folder, domain, shift, path, profile_override))


    if "ogbn_arxiv" in selected_datasets:
        ogb = load_ogbn_arxiv_setting(data_root, profile_override)
        if ogb is not None:
            settings.append(ogb)


    if "twitch" in selected_datasets:
        tw = load_twitch_setting(data_root, profile_override)
        if tw is not None:
            settings.append(tw)

    return settings


def evaluate_logits(logits: Any, bundle: GraphBundle, mask: Any) -> Dict[str, Any]:
    with torch.no_grad():
        mask = mask.bool()
        if int(mask.sum()) <= 0:
            raise ValueError(f"{bundle.name}: empty evaluation mask")

        y_tensor = bundle.y[mask].detach().cpu()
        y = y_tensor.numpy()
        z = logits[mask].detach().cpu()
        pred = z.argmax(dim=-1).numpy()

        acc = float((pred == y).mean())
        micro_f1 = float(f1_score(y, pred, average="micro", zero_division=0))
        macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))


        try:
            ce_loss = float(torch.nn.functional.cross_entropy(z, y_tensor).item())
        except Exception:
            ce_loss = None

        num_classes = int(max(int(bundle.y.max().item()) + 1, z.size(1)))
        counts = np.bincount(pred, minlength=num_classes)
        total_preds = max(1, int(counts.sum()))
        max_fraction = float(counts.max() / total_preds)
        active_classes = int(np.count_nonzero(counts))

        # Argmax concentration is treated as a warning for ranking metrics such as
        # ROC-AUC, while accuracy/F1 tasks use it directly in the hard-collapse guard.
        if num_classes >= 3:
            argmax_concentration = bool(
                max_fraction >= 0.75
                or active_classes <= max(1, num_classes // 2)
            )
        else:
            argmax_concentration = bool(max_fraction >= 0.90 or active_classes <= 1)


        probs_dist = counts[counts > 0] / total_preds
        norm_entropy = float(
            -np.sum(probs_dist * np.log(probs_dist))
            / max(1e-9, np.log(num_classes))
        ) if num_classes > 1 else 1.0

        roc_auc: Optional[float] = None
        ranking_score_mean: Optional[float] = None
        ranking_score_std: Optional[float] = None
        ranking_score_min: Optional[float] = None
        ranking_score_max: Optional[float] = None
        ranking_score_range: Optional[float] = None
        ranking_score_finite = False
        try:
            probs = torch.softmax(z, dim=-1).numpy()
            ranking_score_finite = bool(np.isfinite(probs).all())
            if probs.shape[1] == 2:
                score_matrix = probs[:, 1:2]
                roc_auc = float(roc_auc_score(y, probs[:, 1]))
            else:
                score_matrix = probs
                roc_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))

            # A ranking model is degenerate only when its per-sample scores are
            # effectively constant.  For multiclass probabilities, take the largest
            # across-class sample dispersion/range so a single informative class is
            # enough to establish non-degeneracy.
            class_stds = np.std(score_matrix, axis=0)
            class_ranges = np.ptp(score_matrix, axis=0)
            ranking_score_std = float(np.max(class_stds)) if class_stds.size else 0.0
            ranking_score_range = float(np.max(class_ranges)) if class_ranges.size else 0.0
            ranking_score_mean = float(np.mean(score_matrix))
            ranking_score_min = float(np.min(score_matrix))
            ranking_score_max = float(np.max(score_matrix))
        except Exception:
            roc_auc = None

        return {
            "accuracy": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "roc_auc": roc_auc,
            "loss": ce_loss,
            "argmax_concentration_warning": argmax_concentration,
            "max_predicted_class_fraction": max_fraction,
            "num_predicted_classes": active_classes,
            "total_classes": num_classes,
            "num_true_classes": int(np.unique(y).size),
            "prediction_entropy": norm_entropy,
            "prediction_class_counts": counts.astype(int).tolist(),
            "ranking_score_mean": ranking_score_mean,
            "ranking_score_std": ranking_score_std,
            "ranking_score_min": ranking_score_min,
            "ranking_score_max": ranking_score_max,
            "ranking_score_range": ranking_score_range,
            "ranking_score_finite": ranking_score_finite,
        }


RANKING_COLLAPSE_EPS = 1e-8


def metric_aware_collapse_policy(
    metrics: Mapping[str, Any],
    primary_metric: str,
) -> Dict[str, Any]:
    """Use metric-specific collapse checks; ROC-AUC requires ranking-score degeneracy."""


    out = dict(metrics)
    warning = bool(out.get("argmax_concentration_warning", False))
    metric = str(primary_metric).lower()
    if metric == "roc_auc":
        finite = bool(out.get("ranking_score_finite", False))
        score_std = out.get("ranking_score_std")
        score_range = out.get("ranking_score_range")
        try:
            score_std_f = float(score_std)
            score_range_f = float(score_range)
        except (TypeError, ValueError):
            score_std_f = math.nan
            score_range_f = math.nan
        constant_scores = bool(
            math.isfinite(score_std_f)
            and math.isfinite(score_range_f)
            and score_std_f <= RANKING_COLLAPSE_EPS
            and score_range_f <= RANKING_COLLAPSE_EPS
        )
        hard = bool((not finite) or constant_scores)
        policy = "ranking_score_hard_collapse__argmax_warning_only"
    else:
        hard = warning
        policy = "argmax_concentration_hard_collapse"

    out["argmax_concentration_warning"] = warning
    out["hard_prediction_collapse"] = hard
    out["collapse_policy"] = policy
    out["collapse_policy_primary_metric"] = metric
    return out


def metric_value(metrics: Mapping[str, Any], name: str) -> float:
    val = metrics.get(name)
    if val is None:
        raise RuntimeError(f"Requested metric {name!r} is unavailable in evaluation output")
    return float(val)


def inner_step(
    model: Any,
    optimizer: Any,
    bundle: GraphBundle,
    alpha_div: float,
) -> Dict[str, Any]:
    model.train()
    model.set_inner_loop()
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    augmented = model.forward_adversarial_only(bundle.x, bundle.edge_index)
    loss, parts = tafi.inner_objective(
        logits_by_env=augmented.logits,
        y=bundle.y,
        train_mask=bundle.train_mask,
        alpha_div=float(alpha_div),
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite inner loss")
    loss.backward()
    optimizer.step()
    return scalarize(parts)


def outer_step(
    model: Any,
    predictor_optimizer: Any,
    protection_optimizer: Optional[Any],
    bundle: GraphBundle,
) -> Dict[str, Any]:
    model.train()
    model.set_outer_loop()
    predictor_optimizer.zero_grad(set_to_none=True)
    if protection_optimizer is not None:
        protection_optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    loss, parts = model.outer_objective(
        bundle.x, bundle.edge_index, bundle.y, bundle.train_mask
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite outer loss")
    loss.backward()
    predictor_optimizer.step()
    if protection_optimizer is not None:
        protection_optimizer.step()
    return scalarize(parts)


def build_model(
    train: GraphBundle,
    profile: Mapping[str, Any],
    hyperparams: Mapping[str, Any],
) -> Any:

    num_classes = int(train.y.max().item()) + 1
    hp = resolve_hyperparams(profile, hyperparams)
    dropout = float(hp.get("predictor_dropout", profile.get("predictor_dropout", 0.5)))

    return tafi.TAFI(
        in_dim=int(train.x.size(1)),
        predictor_hidden_dim=int(hp["predictor_hidden_dim"]),
        num_classes=num_classes,
        num_edges=int(train.edge_index.size(1)),
        predictor_num_layers=int(hp["predictor_num_layers"]),
        predictor_dropout=dropout,
        num_envs=int(hp["num_envs"]),
        protection_hidden_dim=int(hp["protection_hidden_dim"]),
        protection_dropout=float(hp["protection_dropout"]),
        perturb_budget_fraction=float(hp["perturb_budget_fraction"]),
        max_edge_attenuation=float(hp["max_edge_attenuation"]),
        protect_eps=float(hp["protect_eps"]),
        priority_floor=float(hp["priority_floor"]),
    )


def _train_one_run_impl(
    *,
    setting: ExperimentSetting,
    seed: int,
    device: Any,
    hyperparams: Mapping[str, Any],
    output_dir: Path,
    log_every: int,
    save_checkpoint: bool,
    save_predictions: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    safe_key = setting.key.replace("/", "__").replace("\\", "__")
    run_dir = output_dir / "runs" / safe_key / "final_model" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_hp = resolve_hyperparams(setting.profile, hyperparams)
    cfg = {
        "setting": setting.key,
        "family": setting.family,
        "domain": setting.domain,
        "shift": setting.shift,
        "seed": seed,
        "profile": setting.profile,
        "hyperparams": resolved_hp,
        "main_metric": setting.main_metric,
    }
    cfg_hash = sha_json(cfg)
    result_path = run_dir / "result.json"

    if not overwrite and result_path.is_file():
        try:
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("run_status") == "COMPLETE" and prior.get("config_sha256") == cfg_hash:
                print(f"[REUSE] Already completed: {setting.key} seed={seed}", flush=True)
                return prior
        except Exception:
            pass

    write_json(run_dir / "config.json", cfg)
    seed_everything(seed)

    train_cpu = setting.train
    val_cpu = setting.val
    test_cpu = setting.tests

    train = move_graph(train_cpu, device)
    val = move_graph(val_cpu, device)
    tests = [(name, move_graph(g, device)) for name, g in test_cpu]

    model = build_model(train_cpu, setting.profile, hyperparams).to(device)

    predictor_opt = torch.optim.AdamW(
        model.predictor_parameters(),
        lr=float(setting.profile["lr_predictor"]),
        weight_decay=float(resolved_hp["predictor_weight_decay"]),
    )
    augmenter_opt = torch.optim.AdamW(
        model.augmenter_parameters(),
        lr=float(resolved_hp["lr_augmenter"]),
        weight_decay=float(resolved_hp["augmenter_weight_decay"]),
    )
    protection_opt = torch.optim.AdamW(
        model.protection_generator_parameters(),
        lr=float(resolved_hp["lr_protection_generator"]),
        weight_decay=float(resolved_hp["protection_generator_weight_decay"]),
    )

    selection_metric = str(setting.profile["selection_metric"])
    collapse_guard = bool(setting.profile["collapse_guard"])
    epochs = int(setting.profile["epochs"])
    patience = int(setting.profile["patience"])
    inner_steps_count = int(resolved_hp["inner_steps"])
    alpha_div = float(resolved_hp["alpha_div"])

    best_safe_score = -math.inf
    best_safe_epoch = -1
    best_safe_state: Optional[Dict[str, Any]] = None
    stale = 0
    start_time = time.perf_counter()

    with (run_dir / "training_log.jsonl").open("w", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            inner_records = [
                inner_step(model, augmenter_opt, train, alpha_div)
                for _ in range(inner_steps_count)
            ]
            outer_record = outer_step(model, predictor_opt, protection_opt, train)

            model.eval()
            with torch.no_grad():
                val_logits = model.forward_source_output(val.x, val.edge_index).logits
                val_metrics = metric_aware_collapse_policy(
                    evaluate_logits(val_logits, val, val.val_mask),
                    selection_metric,
                )

            score = metric_value(val_metrics, selection_metric)
            argmax_warning = bool(val_metrics["argmax_concentration_warning"])
            hard_collapsed = bool(val_metrics["hard_prediction_collapse"])
            eligible = bool(not collapse_guard or not hard_collapsed)

            if eligible and score > best_safe_score:
                best_safe_score = score
                best_safe_epoch = epoch + 1
                best_safe_state = clone_state(model)
                stale = 0
            else:
                stale += 1

            rec = {
                "epoch": epoch + 1,
                "setting": setting.key,
                "seed": seed,
                "selection_metric": selection_metric,
                "val_score": score,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_hard_collapsed": hard_collapsed,
                "val_argmax_concentration_warning": argmax_warning,
                "val_collapse_policy": val_metrics["collapse_policy"],
                "best_safe_epoch": best_safe_epoch,
                "best_safe_score": best_safe_score,
                "loss_inner": mean([float(r["loss_inner"]) for r in inner_records]),
                "inner_loss_cls": mean([float(r["loss_cls"]) for r in inner_records]),
                "inner_loss_div": mean([float(r["loss_div"]) for r in inner_records]),
                "inner_environment_risk_variance": mean([float(r["environment_risk_variance"]) for r in inner_records]),
                "loss_outer": float(outer_record["loss_outer"]),
                "outer_loss_cls": float(outer_record["loss_cls"]),
                "outer_environment_risk_variance": float(outer_record["environment_risk_variance"]),
            }
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()

            if epoch == 0 or (epoch + 1) % log_every == 0 or (epoch + 1) == epochs:
                print(
                    f"[{setting.key} | seed {seed}] Ep {epoch+1:03d}/{epochs:03d} | "
                    f"Val {selection_metric}: {score:.4f} (best: {best_safe_score:.4f} @ep{best_safe_epoch}) | "
                    f"HardCollapse: {hard_collapsed} | ArgmaxWarn: {argmax_warning}",
                    flush=True,
                )

            if stale >= patience:
                print(f"[{setting.key} | seed {seed}] Early stopping triggered at epoch {epoch+1}.", flush=True)
                break

    if best_safe_state is None:
        raise RuntimeError(f"No eligible validation checkpoint found for {setting.key} seed={seed}")

    model.load_state_dict(best_safe_state, strict=True)
    model.eval()

    with torch.no_grad():
        train_logits = model.forward_source_output(train.x, train.edge_index).logits
        selected_train = metric_aware_collapse_policy(
            evaluate_logits(train_logits, train, train.train_mask),
            selection_metric,
        )
        val_logits = model.forward_source_output(val.x, val.edge_index).logits
        selected_val = metric_aware_collapse_policy(
            evaluate_logits(val_logits, val, val.val_mask),
            selection_metric,
        )

    with torch.no_grad():
        _, selected_outer_parts = model.outer_objective(
            train.x, train.edge_index, train.y, train.train_mask
        )
    selected_outer_parts = scalarize(selected_outer_parts)

    test_metrics: Dict[str, Dict[str, Any]] = {}
    output_payload: Dict[str, Any] = {
        "train_predictions": train_logits.argmax(dim=-1).detach().cpu(),
        "val_predictions": val_logits.argmax(dim=-1).detach().cpu(),
    }

    for test_name, test_graph in tests:
        with torch.no_grad():
            logits = model.forward_source_output(test_graph.x, test_graph.edge_index).logits
            t_metrics = metric_aware_collapse_policy(
                evaluate_logits(logits, test_graph, test_graph.test_mask),
                setting.main_metric,
            )
        test_metrics[test_name] = t_metrics
        output_payload[f"test_predictions__{test_name}"] = logits.argmax(dim=-1).detach().cpu()

    primary_values = [metric_value(m, setting.main_metric) for m in test_metrics.values()]
    acc_values = [float(m["accuracy"]) for m in test_metrics.values()]
    f1_values = [float(m["macro_f1"]) for m in test_metrics.values()]
    micro_f1_values = [float(m["micro_f1"]) for m in test_metrics.values()]
    auc_values = [float(m["roc_auc"]) for m in test_metrics.values() if m["roc_auc"] is not None]
    test_losses = [float(m["loss"]) for m in test_metrics.values() if m.get("loss") is not None]
    max_fractions = [float(m["max_predicted_class_fraction"]) for m in test_metrics.values() if m.get("max_predicted_class_fraction") is not None]
    active_cls_list = [float(m["num_predicted_classes"]) for m in test_metrics.values() if m.get("num_predicted_classes") is not None]
    entropies = [float(m["prediction_entropy"]) for m in test_metrics.values() if m.get("prediction_entropy") is not None]

    elapsed = time.perf_counter() - start_time
    test_acc_mean = mean(acc_values)
    train_acc = float(selected_train["accuracy"])
    val_acc = float(selected_val["accuracy"])

    result: Dict[str, Any] = {
        "setting": setting.key,
        "family": setting.family,
        "domain": setting.domain,
        "shift": setting.shift,
        "seed": seed,
        "main_metric": setting.main_metric,
        

        "train_accuracy": train_acc,
        "train_macro_f1": float(selected_train["macro_f1"]),
        "train_loss": selected_train.get("loss"),


        "best_epoch": best_safe_epoch,
        "best_val_score": best_safe_score,
        "best_val_accuracy": val_acc,
        "best_val_macro_f1": float(selected_val["macro_f1"]),
        "best_val_roc_auc": selected_val.get("roc_auc"),
        "best_val_loss": selected_val.get("loss"),
        "best_val_hard_collapse": bool(selected_val.get("hard_prediction_collapse", False)),
        "best_val_argmax_concentration_warning": bool(selected_val.get("argmax_concentration_warning", False)),
        "best_val_ranking_score_std": selected_val.get("ranking_score_std"),
        "best_val_ranking_score_range": selected_val.get("ranking_score_range"),
        "collapse_policy": selected_val.get("collapse_policy"),


        "test_domains": test_metrics,
        "test_primary_mean": mean(primary_values),
        "test_accuracy_mean": test_acc_mean,
        "test_micro_f1_mean": mean(micro_f1_values),
        "test_macro_f1_mean": mean(f1_values),
        "test_roc_auc_mean": mean(auc_values) if auc_values else None,
        "test_loss_mean": mean(test_losses) if test_losses else None,


        "generalization_gap": train_acc - test_acc_mean,
        "val_test_gap": val_acc - test_acc_mean,


        "test_max_class_fraction": mean(max_fractions) if max_fractions else None,
        "test_active_classes": mean(active_cls_list) if active_cls_list else None,
        "test_prediction_entropy": mean(entropies) if entropies else None,
        "test_any_hard_collapse": any(bool(m.get("hard_prediction_collapse", False)) for m in test_metrics.values()),
        "test_any_argmax_concentration_warning": any(bool(m.get("argmax_concentration_warning", False)) for m in test_metrics.values()),


        "num_envs": int(model.num_envs),
        "inner_steps": inner_steps_count,
        "alpha_div": alpha_div,
        "allocation_method": "capped_proportional_fixed_mass",
        "protect_eps": float(resolved_hp["protect_eps"]),
        "priority_floor": float(resolved_hp["priority_floor"]),
        "selected_loss_outer": float(selected_outer_parts["loss_outer"]),
        "selected_loss_cls": float(selected_outer_parts["loss_cls"]),
        "selected_environment_risk_variance": float(selected_outer_parts["environment_risk_variance"]),


        "epochs_trained": epoch + 1,
        "early_stopped": bool(stale >= patience),
        "stale_epochs_at_best": (epoch + 1) - best_safe_epoch,
        "runtime_seconds": elapsed,
        "config_sha256": cfg_hash,
        "run_status": "COMPLETE",
        "run_directory": str(run_dir),
    }

    if save_checkpoint:
        torch.save(
            {
                "state_dict": best_safe_state,
                "best_epoch": best_safe_epoch,
                "best_val_score": best_safe_score,
                "selection_metric": selection_metric,
                "config": cfg,
            },
            run_dir / "checkpoint.pt",
        )

    if save_predictions:
        torch.save(output_payload, run_dir / "final_outputs.pt")

    write_json(result_path, result)

    print(
        f"[FINISHED {setting.key} | seed {seed}] Test {setting.main_metric}: {result['test_primary_mean']:.4f} | "
        f"Acc: {result['test_accuracy_mean']:.4f} | F1: {result['test_macro_f1_mean']:.4f} | Time: {elapsed:.1f}s\n",
        flush=True,
    )
    return result


def _clear_setting_cuda_cache() -> None:


    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def train_one_run(
    *,
    setting: ExperimentSetting,
    seed: int,
    device: Any,
    hyperparams: Mapping[str, Any],
    output_dir: Path,
    log_every: int,
    save_checkpoint: bool,
    save_predictions: bool,
    overwrite: bool,
) -> Dict[str, Any]:

    try:
        return _train_one_run_impl(
            setting=setting,
            seed=seed,
            device=device,
            hyperparams=hyperparams,
            output_dir=output_dir,
            log_every=log_every,
            save_checkpoint=save_checkpoint,
            save_predictions=save_predictions,
            overwrite=overwrite,
        )
    finally:
        _clear_setting_cuda_cache()


def _classify_run_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "no eligible validation checkpoint" in text or "collapse" in text:
        return "FAILED_COLLAPSE"
    if "out of memory" in text or "cuda oom" in text:
        return "FAILED_OOM"
    if any(token in text for token in ("nan", "inf", "non-finite", "nonfinite")):
        return "FAILED_NUMERICAL"
    return "FAILED_EXCEPTION"


def _failure_result(
    *,
    setting: ExperimentSetting,
    seed: int,
    hyperparams: Mapping[str, Any],
    output_dir: Path,
    exc: BaseException,
) -> Dict[str, Any]:

    safe_key = setting.key.replace("/", "__").replace("\\", "__")
    run_dir = output_dir / "runs" / safe_key / "final_model" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_hyperparams(setting.profile, hyperparams)
    status = _classify_run_failure(exc)
    reason = f"{type(exc).__name__}: {exc}"
    (run_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    row: Dict[str, Any] = {
        "setting": setting.key,
        "family": setting.family,
        "domain": setting.domain,
        "shift": setting.shift,
        "seed": seed,
        "main_metric": setting.main_metric,
        "run_status": status,
        "failure_reason": reason,
        "run_directory": str(run_dir),
    }
    write_json(run_dir / "result.json", row)
    return row


def aggregate_results(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["setting"], []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    t1_lines: List[str] = [
        "### 1. Test Performance Overview",
        "| Setting | Main Metric | Seeds (N) | Test Primary (Mean±Std) | Test Accuracy | Test Macro-F1 | Test ROC-AUC | Runtime (avg) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    t2_lines: List[str] = [
        "### 2. Training, Validation, and Test Metrics",
        "| Setting | Train Acc | Best Val Score | Test Acc | Train-Test Accuracy Gap | Val-Test Accuracy Gap |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    t3_lines: List[str] = [
        "### 3. Training Progress and Early Stopping",
        "| Setting | Best Epoch (Mean±Std) | Trained Epochs (Avg) | Early-Stopped Runs | Checkpoint Timing |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    t4_lines: List[str] = [
        "### 4. Classifier Health & Distribution Balance",
        "| Setting | Top-1 Class % | Active Classes | Pred Entropy | Hard Collapse | Argmax Warn | Health Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    warnings: List[str] = []

    for key, grp in sorted(grouped.items()):
        metric = grp[0]["main_metric"]
        primary = [float(r["test_primary_mean"]) for r in grp if r.get("test_primary_mean") is not None]
        acc = [float(r["test_accuracy_mean"]) for r in grp if r.get("test_accuracy_mean") is not None]
        f1 = [float(r["test_macro_f1_mean"]) for r in grp if r.get("test_macro_f1_mean") is not None]
        aucs = [float(r["test_roc_auc_mean"]) for r in grp if r.get("test_roc_auc_mean") is not None]
        train_accs = [float(r["train_accuracy"]) for r in grp if r.get("train_accuracy") is not None]
        val_scores = [float(r["best_val_score"]) for r in grp if r.get("best_val_score") is not None]
        gen_gaps = [float(r["generalization_gap"]) for r in grp if r.get("generalization_gap") is not None]
        val_gaps = [float(r["val_test_gap"]) for r in grp if r.get("val_test_gap") is not None]
        best_eps = [float(r["best_epoch"]) for r in grp if r.get("best_epoch") is not None]
        epochs_tr = [float(r["epochs_trained"]) for r in grp if r.get("epochs_trained") is not None]
        top1_fracs = [float(r["test_max_class_fraction"]) for r in grp if r.get("test_max_class_fraction") is not None]
        entropies = [float(r["test_prediction_entropy"]) for r in grp if r.get("test_prediction_entropy") is not None]
        active_cls = [float(r["test_active_classes"]) for r in grp if r.get("test_active_classes") is not None]
        times = [float(r["runtime_seconds"]) for r in grp if r.get("runtime_seconds") is not None]
        hard_collapse_cnt = sum(
            bool(r.get("test_any_hard_collapse", False)) for r in grp
        )
        argmax_warning_cnt = sum(
            bool(r.get("test_any_argmax_concentration_warning", False)) for r in grp
        )
        early_stopped_cnt = sum(bool(r.get("early_stopped", False)) for r in grp)

        m_prim, s_prim = mean(primary), sample_std(primary)
        m_acc, s_acc = mean(acc), sample_std(acc)
        m_f1, s_f1 = mean(f1), sample_std(f1)
        m_auc_str = f"{mean(aucs):.4f}±{sample_std(aucs):.4f}" if aucs else "N/A"
        avg_time = f"{mean(times):.1f}s"

        m_tr_acc = mean(train_accs)
        m_val_sc = mean(val_scores)
        m_gen_gap = mean(gen_gaps)
        m_val_gap = mean(val_gaps)
        m_bep = mean(best_eps)
        s_bep = sample_std(best_eps)
        m_etr = mean(epochs_tr)
        m_top1 = mean(top1_fracs)
        m_ent = mean(entropies)
        m_act_cls = mean(active_cls)


        if m_bep <= 5.0:
            conv_status = "Best checkpoint in first 5 epochs"
            warnings.append(f"- **{key}**: Best validation score occurs very early ({m_bep:.1f} epochs); inspect the saved training log if this is unexpected.")
        elif m_bep >= m_etr - 3.0:
            conv_status = "Best checkpoint near training limit"
            warnings.append(f"- **{key}**: Best validation score occurs near the training horizon ({m_bep:.1f}/{m_etr:.1f}).")
        else:
            conv_status = "Best checkpoint away from boundaries"

        if hard_collapse_cnt > 0:
            health_status = f"❌ Hard Collapse ({hard_collapse_cnt}/{len(grp)} runs)"
            warnings.append(
                f"- **{key}**: Hard collapse detected under the primary-metric-aware policy "
                f"({hard_collapse_cnt}/{len(grp)} runs)."
            )
        elif str(metric).lower() == "roc_auc" and argmax_warning_cnt > 0:
            health_status = f"⚠️ Argmax Concentration Warning ({argmax_warning_cnt}/{len(grp)})"
            warnings.append(
                f"- **{key}**: ROC-AUC remains the primary health signal; "
                f"{argmax_warning_cnt}/{len(grp)} runs show high argmax concentration "
                f"(mean Top-1 class = {m_top1*100:.1f}%). This warning does not imply hard collapse."
            )
        elif m_top1 >= 0.70:
            health_status = "⚠️ High Prediction Concentration"
        elif m_top1 >= 0.50:
            health_status = "⚠️ Moderately Imbalanced"
        else:
            health_status = "✅ Balanced"

        summary_rows.append({
            "setting": key,
            "family": grp[0]["family"],
            "domain": grp[0]["domain"],
            "shift": grp[0]["shift"],
            "main_metric": metric,
            "n_seeds": len(grp),
            "mean_test_primary": m_prim,
            "std_test_primary": s_prim,
            "mean_test_accuracy": m_acc,
            "std_test_accuracy": s_acc,
            "mean_test_macro_f1": m_f1,
            "std_test_macro_f1": s_f1,
            "mean_test_roc_auc": mean(aucs) if aucs else None,
            "std_test_roc_auc": sample_std(aucs) if aucs else None,
            "mean_train_accuracy": m_tr_acc,
            "mean_best_val_score": m_val_sc,
            "mean_generalization_gap": m_gen_gap,
            "mean_val_test_gap": m_val_gap,
            "mean_best_epoch": m_bep,
            "std_best_epoch": s_bep,
            "mean_epochs_trained": m_etr,
            "early_stopped_count": early_stopped_cnt,
            "mean_top1_class_fraction": m_top1,
            "mean_active_classes": m_act_cls,
            "mean_prediction_entropy": m_ent,
            "collapse_run_count": hard_collapse_cnt,
            "hard_collapse_run_count": hard_collapse_cnt,
            "argmax_concentration_warning_run_count": argmax_warning_cnt,
            "avg_runtime_seconds": mean(times),
        })

        t1_lines.append(
            f"| **{key}** | {metric} | {len(grp)} | **{m_prim:.4f} ± {s_prim:.4f}** | "
            f"{m_acc:.4f} ± {s_acc:.4f} | {m_f1:.4f} ± {s_f1:.4f} | {m_auc_str} | {avg_time} |"
        )
        t2_lines.append(
            f"| **{key}** | {m_tr_acc:.4f} | {m_val_sc:.4f} | {m_acc:.4f} | "
            f"{'+' if m_gen_gap>=0 else ''}{m_gen_gap:.4f} | {'+' if m_val_gap>=0 else ''}{m_val_gap:.4f} |"
        )
        t3_lines.append(
            f"| **{key}** | {m_bep:.1f} ± {s_bep:.1f} | {m_etr:.1f} | {early_stopped_cnt}/{len(grp)} | {conv_status} |"
        )
        t4_lines.append(
            f"| **{key}** | {m_top1*100:.1f}% | {m_act_cls:.1f} | {m_ent:.3f} | "
            f"{hard_collapse_cnt}/{len(grp)} | {argmax_warning_cnt}/{len(grp)} | {health_status} |"
        )

    lines: List[str] = [
        "# TAFI Benchmark Summary",
        "",
        *t1_lines,
        "",
        *t2_lines,
        "",
        *t3_lines,
        "",
        *t4_lines,
        "",
        "### 5. Run Health Notes",
    ]
    if warnings:
        lines.extend(warnings)
    else:
        lines.append("- No checkpoint-timing or prediction-concentration warnings were triggered.")
    lines.append("")

    report_md = "\n".join(lines) + "\n"
    return summary_rows, report_md


ELLIPTIC_TRAIN_SNAPSHOTS = list(range(6, 11))
ELLIPTIC_VAL_SNAPSHOTS = list(range(11, 16))
ELLIPTIC_TEST_SNAPSHOTS = list(range(16, 49))
ELLIPTIC_TEST_GROUPS = [
    ("16", [16]),
    ("17-20", list(range(17, 21))),
    ("21-24", list(range(21, 25))),
    ("25-28", list(range(25, 29))),
    ("29-32", list(range(29, 33))),
    ("33-36", list(range(33, 37))),
    ("37-40", list(range(37, 41))),
    ("41-44", list(range(41, 45))),
    ("45-48", list(range(45, 49))),
]


def _elliptic_snapshot_path(data_root: Path, idx: int) -> Path:
    for name in (f"{idx}.pkl", f"{idx:02d}.pkl"):
        candidate = data_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing Elliptic snapshot {idx} under {data_root}")


def _elliptic_load_snapshot(data_root: Path, idx: int) -> GraphBundle:
    import pickle

    path = _elliptic_snapshot_path(data_root, idx)
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        raise ValueError(f"Unexpected Elliptic snapshot format for {idx}: {type(obj)}")
    adjacency, labels, features = obj
    rows, cols = adjacency.nonzero()
    edge_index = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    if hasattr(features, "toarray"):
        feature_array = features.toarray()
    else:
        feature_array = np.asarray(features)
    feature_array = np.asarray(feature_array, dtype=np.float32)
    x = torch.as_tensor(feature_array, dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels).reshape(-1), dtype=torch.long)
    labeled = y >= 0
    return GraphBundle(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=labeled.clone(),
        val_mask=labeled.clone(),
        test_mask=labeled.clone(),
        name=f"Elliptic/{idx}",
        source_path=str(path),
    )


def _elliptic_fit_scaler(
    snapshots: Dict[int, GraphBundle], fit_indices: Sequence[int], scaler_name: str
) -> Dict[str, Any]:
    if scaler_name == "none":
        return {"name": "none"}
    if scaler_name in {"row_standard", "row_l2", "snapshot_standard"}:
        for graph in snapshots.values():
            if scaler_name == "row_standard":
                center = graph.x.mean(dim=1, keepdim=True)
                scale = graph.x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            elif scaler_name == "row_l2":
                center = torch.zeros_like(graph.x.mean(dim=1, keepdim=True))
                scale = graph.x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
            else:
                center = graph.x.mean(dim=0)
                scale = graph.x.std(dim=0, unbiased=False).clamp_min(1e-6)
            graph.x = (graph.x - center) / scale
        return {"name": scaler_name, "fit_snapshots": "all_snapshots"}
    fit_parts = [snapshots[i].x[snapshots[i].train_mask.bool()] for i in fit_indices]
    if not fit_parts:
        raise ValueError("Cannot fit Elliptic scaler without labeled train nodes")
    fit_x = torch.cat(fit_parts, dim=0)
    if scaler_name == "standard":
        center = fit_x.mean(dim=0)
        scale = fit_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    elif scaler_name == "minmax":
        center = fit_x.min(dim=0).values
        scale = (fit_x.max(dim=0).values - center).clamp_min(1e-6)
    else:
        raise ValueError(f"Unknown Elliptic scaler: {scaler_name}")
    for graph in snapshots.values():
        graph.x = (graph.x - center) / scale
    return {
        "name": scaler_name,
        "fit_snapshots": list(fit_indices),
        "fit_nodes": int(fit_x.size(0)),
    }


def _elliptic_pool(snapshots: Sequence[GraphBundle]) -> GraphBundle:
    if not snapshots:
        raise ValueError("Cannot pool an empty Elliptic snapshot sequence")
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    train_masks: List[torch.Tensor] = []
    val_masks: List[torch.Tensor] = []
    test_masks: List[torch.Tensor] = []
    edges: List[torch.Tensor] = []
    offset = 0
    for graph in snapshots:
        xs.append(graph.x)
        ys.append(graph.y)
        train_masks.append(graph.train_mask)
        val_masks.append(graph.val_mask)
        test_masks.append(graph.test_mask)
        edges.append(graph.edge_index + offset)
        offset += graph.x.size(0)
    return GraphBundle(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edges, dim=1),
        y=torch.cat(ys, dim=0),
        train_mask=torch.cat(train_masks, dim=0),
        val_mask=torch.cat(val_masks, dim=0),
        test_mask=torch.cat(test_masks, dim=0),
        name="Elliptic/pooled",
        source_path="pooled_disconnected_snapshots",
    )


def _elliptic_macro_f1(logits: torch.Tensor, graph: GraphBundle) -> float:
    mask = graph.test_mask.bool() & (graph.y >= 0)
    if int(mask.sum()) == 0:
        return float("nan")
    target = graph.y[mask].detach().cpu().numpy()
    pred = logits[mask].argmax(dim=-1).detach().cpu().numpy()
    return float(f1_score(target, pred, average="macro", zero_division=0))


def _elliptic_predict(
    model: Any, graph: GraphBundle, device: Any
) -> torch.Tensor:
    moved = move_graph(graph, device)
    with torch.no_grad():
        model.eval()
        logits = model.forward_source_output(moved.x, moved.edge_index).logits
        return logits.detach().cpu()


def _elliptic_group_f1(
    period_logits: Mapping[str, torch.Tensor],
    test_graphs: Mapping[int, GraphBundle],
    indices: Sequence[int],
) -> float:
    targets: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    for idx in indices:
        graph = test_graphs[idx]
        mask = graph.test_mask.bool() & (graph.y >= 0)
        targets.append(graph.y[mask].detach().cpu().numpy())
        predictions.append(period_logits[str(idx)][mask].argmax(dim=-1).detach().cpu().numpy())
    return float(
        f1_score(
            np.concatenate(targets),
            np.concatenate(predictions),
            average="macro",
            zero_division=0,
        )
    )


def _elliptic_clone_state(models: Sequence[Any]) -> Dict[str, Any]:
    return {
        "predictor": clone_state(models[0].predictor),
        "protection_generator": clone_state(models[0].protection_generator),
        "augmenters": [clone_state(model.augmenter) for model in models],
    }


def _elliptic_load_state(models: Sequence[Any], state: Mapping[str, Any]) -> None:
    models[0].predictor.load_state_dict(state["predictor"])
    models[0].protection_generator.load_state_dict(state["protection_generator"])
    for model, augmenter_state in zip(models, state["augmenters"]):
        model.augmenter.load_state_dict(augmenter_state)


def _elliptic_profile(args: Any) -> Dict[str, Any]:

    profile = dict(DEFAULT_MODEL_HYPERPARAMS)
    profile.update(get_profile_for_setting("Elliptic"))
    profile.update(
        {key: value for key, value in get_hyperparams(args).items() if value is not None}
    )
    if args.scaler != "auto":
        profile["scaler"] = args.scaler
    if args.predictor_dropout is not None:
        profile["predictor_dropout"] = args.predictor_dropout
    if args.lr_predictor is not None:
        profile["lr_predictor"] = args.lr_predictor
    if args.epochs is not None:
        profile["epochs"] = args.epochs
    if args.patience is not None:
        profile["patience"] = args.patience
    return profile


def _elliptic_snapshotwise_seed_impl(
    seed: int,
    train_graphs: Sequence[GraphBundle],
    train_graph: GraphBundle,
    val_graphs: Sequence[GraphBundle],
    test_graphs: Mapping[int, GraphBundle],
    profile: Mapping[str, Any],
    device: Any,
) -> Dict[str, Any]:
    seed_everything(seed)
    start_time = time.perf_counter()
    train = [move_graph(graph, device) for graph in train_graphs]
    val = [move_graph(graph, device) for graph in val_graphs]
    val_pool = _elliptic_pool(val)
    hyperparams = dict(DEFAULT_MODEL_HYPERPARAMS)
    hyperparams.update(profile)

    models = [build_model(graph, profile, hyperparams).to(device) for graph in train]
    shared_predictor = models[0].predictor
    shared_protection_generator = models[0].protection_generator
    for model in models[1:]:
        model.predictor = shared_predictor
        model.protection_generator = shared_protection_generator

    predictor_optimizer = torch.optim.AdamW(
        shared_predictor.parameters(),
        lr=float(profile["lr_predictor"]),
        weight_decay=float(hyperparams["predictor_weight_decay"]),
    )
    augmenter_optimizer = torch.optim.AdamW(
        [parameter for model in models for parameter in model.augmenter_parameters()],
        lr=float(hyperparams["lr_augmenter"]),
        weight_decay=float(hyperparams["augmenter_weight_decay"]),
    )
    protection_optimizer = torch.optim.AdamW(
        shared_protection_generator.parameters(),
        lr=float(hyperparams["lr_protection_generator"]),
        weight_decay=float(hyperparams["protection_generator_weight_decay"]),
    )

    best_any_score = -float("inf")
    best_safe_score = -float("inf")
    best_any_state = None
    best_safe_state = None
    best_any_epoch = -1
    best_safe_epoch = -1
    stale = 0
    guard_fallback = False
    training_log: List[Dict[str, Any]] = []

    for epoch in range(int(profile["epochs"])):
        inner_records: List[Mapping[str, Any]] = []
        outer_records: List[Mapping[str, Any]] = []
        for model, graph in zip(models, train):
            model.train()
            for _ in range(int(profile["inner_steps"])):
                inner_records.append(inner_step(model, augmenter_optimizer, graph, float(profile["alpha_div"])))
            outer_records.append(outer_step(model, predictor_optimizer, protection_optimizer, graph))

        model = models[0]
        val_logits = _elliptic_predict(model, val_pool, device)
        # ``_elliptic_predict`` returns CPU logits, so evaluate with CPU labels
        # and masks even when the training pool was moved to CUDA.
        val_y_cpu = val_pool.y.detach().cpu()
        val_mask = val_pool.val_mask.detach().cpu().bool() & (val_y_cpu >= 0)
        val_pred = val_logits.argmax(dim=-1)
        val_active = int(torch.unique(val_pred[val_mask]).numel()) if int(val_mask.sum()) else 0
        val_top_fraction = float(torch.bincount(val_pred[val_mask], minlength=2).max() / val_mask.sum()) if int(val_mask.sum()) else 1.0
        val_snapshot_scores = []
        for graph in val_graphs:
            val_snapshot_scores.append(_elliptic_macro_f1(_elliptic_predict(model, graph, device), graph))
        val_f1 = float(np.mean(np.asarray(val_snapshot_scores, dtype=float)))
        collapsed = bool(val_active <= 1 or val_top_fraction >= 0.90)
        training_log.append(
            {
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "active_classes": val_active,
                "top_fraction": val_top_fraction,
                "collapsed": collapsed,
                "loss_inner": float(np.mean([float(r["loss_inner"]) for r in inner_records])),
                "inner_loss_cls": float(np.mean([float(r["loss_cls"]) for r in inner_records])),
                "inner_loss_div": float(np.mean([float(r["loss_div"]) for r in inner_records])),
                "inner_environment_risk_variance": float(np.mean([float(r["environment_risk_variance"]) for r in inner_records])),
                "loss_outer": float(np.mean([float(r["loss_outer"]) for r in outer_records])),
                "outer_loss_cls": float(np.mean([float(r["loss_cls"]) for r in outer_records])),
                "outer_environment_risk_variance": float(np.mean([float(r["environment_risk_variance"]) for r in outer_records])),
            }
        )

        improved_any = val_f1 > best_any_score
        improved_safe = (not collapsed) and val_f1 > best_safe_score
        if improved_any:
            best_any_score = val_f1
            best_any_state = _elliptic_clone_state(models)
            best_any_epoch = epoch
        if improved_safe:
            best_safe_score = val_f1
            best_safe_state = _elliptic_clone_state(models)
            best_safe_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(profile["patience"]):
            break

    if best_safe_state is not None:
        _elliptic_load_state(models, best_safe_state)
        selected_epoch, selected_val = best_safe_epoch, best_safe_score
    elif best_any_state is not None:
        _elliptic_load_state(models, best_any_state)
        selected_epoch, selected_val = best_any_epoch, best_any_score
        guard_fallback = True
    else:
        raise RuntimeError(f"No Elliptic checkpoint was produced for seed {seed}")

    selected_log = training_log[selected_epoch]
    model = models[0]
    train_logits = _elliptic_predict(model, train_graph, device)
    train_f1 = _elliptic_macro_f1(train_logits, train_graph)
    period_logits = {
        str(idx): _elliptic_predict(model, test_graphs[idx], device)
        for idx in ELLIPTIC_TEST_SNAPSHOTS
    }
    period_f1 = {
        str(idx): _elliptic_macro_f1(period_logits[str(idx)], test_graphs[idx])
        for idx in ELLIPTIC_TEST_SNAPSHOTS
    }
    group_f1 = {
        name: _elliptic_group_f1(period_logits, test_graphs, indices)
        for name, indices in ELLIPTIC_TEST_GROUPS
    }
    return {
        "seed": seed,
        "epochs_ran": len(training_log),
        "selected_epoch": selected_epoch,
        "train_macro_f1": train_f1,
        "best_val_macro_f1": selected_val,
        "overall_test_macro_f1": _elliptic_group_f1(period_logits, test_graphs, ELLIPTIC_TEST_SNAPSHOTS),
        "period_f1": period_f1,
        "group_f1": group_f1,
        "guard_fallback": guard_fallback,
        "runtime_seconds": time.perf_counter() - start_time,
        "run_status": "COMPLETE",
        "allocation_method": "capped_proportional_fixed_mass",
        "selected_loss_outer": float(selected_log["loss_outer"]),
        "selected_loss_cls": float(selected_log["outer_loss_cls"]),
        "selected_environment_risk_variance": float(selected_log["outer_environment_risk_variance"]),
        "training_log": training_log,
        "_tensor_artifacts": {
            "test_logits": period_logits,
            "test_labels": {str(idx): test_graphs[idx].y.detach().cpu() for idx in ELLIPTIC_TEST_SNAPSHOTS},
        },
    }


def _elliptic_snapshotwise_seed(
    seed: int,
    train_graphs: Sequence[GraphBundle],
    train_graph: GraphBundle,
    val_graphs: Sequence[GraphBundle],
    test_graphs: Mapping[int, GraphBundle],
    profile: Mapping[str, Any],
    device: Any,
) -> Dict[str, Any]:

    try:
        return _elliptic_snapshotwise_seed_impl(
            seed, train_graphs, train_graph, val_graphs, test_graphs, profile, device
        )
    finally:
        _clear_setting_cuda_cache()


def _run_elliptic(args: Any, output_dir: Path) -> None:
    data_root = Path(args.data_root).resolve()
    nested = data_root / "Elliptic"
    if nested.is_dir():
        data_root = nested
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    profile = _elliptic_profile(args)
    snapshots = {idx: _elliptic_load_snapshot(data_root, idx) for idx in range(49)}
    scaler_stats = _elliptic_fit_scaler(snapshots, ELLIPTIC_TRAIN_SNAPSHOTS, str(profile["scaler"]))
    train_graphs = [snapshots[idx] for idx in ELLIPTIC_TRAIN_SNAPSHOTS]
    val_graphs = [snapshots[idx] for idx in ELLIPTIC_VAL_SNAPSHOTS]
    test_graphs = {idx: snapshots[idx] for idx in ELLIPTIC_TEST_SNAPSHOTS}
    train_graph = _elliptic_pool(train_graphs)
    manifest = {
        "dataset": "Elliptic",
        "protocol_source": "SING elliptic/v_main_elliptic.py",
        "train_snapshots": ELLIPTIC_TRAIN_SNAPSHOTS,
        "validation_snapshots": ELLIPTIC_VAL_SNAPSHOTS,
        "test_snapshots": ELLIPTIC_TEST_SNAPSHOTS,
        "test_groups": {name: indices for name, indices in ELLIPTIC_TEST_GROUPS},
        "metric": "macro_f1",
        "seeds": seeds,
        "device": str(device),
        "data_root": str(data_root),
        "train_mode": "snapshotwise",
        "feature_scaler": scaler_stats,
        "profile": profile,
    }
    write_json(output_dir / "manifest.json", manifest)

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"[Elliptic | seed {seed}] snapshotwise training", flush=True)
        try:
            result = _elliptic_snapshotwise_seed(
                seed, train_graphs, train_graph, val_graphs, test_graphs, profile, device
            )
            if bool(result.get("guard_fallback", False)):
                result["run_status"] = "FAILED_COLLAPSE"
                result["failure_reason"] = (
                    "Elliptic collapse guard fallback: no non-collapsed validation "
                    "checkpoint was available."
                )
            tensor_artifacts = result.pop("_tensor_artifacts")
            torch.save(tensor_artifacts, output_dir / f"seed_{seed}_outputs.pt")
        except Exception as exc:
            result = {
                "seed": seed,
                "run_status": _classify_run_failure(exc),
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
            (output_dir / f"seed_{seed}_failure_traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        results.append(result)
        write_json(output_dir / f"seed_{seed}_result.json", result)

    all_rows = []
    period_rows = []
    group_rows = []
    for result in results:
        if str(result.get("run_status", "COMPLETE")) != "COMPLETE":
            all_rows.append(
                {
                    "setting": "Elliptic/time/temporal",
                    "family": "Elliptic",
                    "domain": "temporal",
                    "shift": "snapshotwise",
                    "seed": result["seed"],
                    "main_metric": "macro_f1",
                    "run_status": result.get("run_status", "FAILED_EXCEPTION"),
                    "failure_reason": result.get("failure_reason", ""),
                }
            )
            continue
        all_rows.append(
            {
                "setting": "Elliptic/time/temporal",
                "family": "Elliptic",
                "domain": "temporal",
                "shift": "snapshotwise",
                "seed": result["seed"],
                "main_metric": "macro_f1",
                "best_epoch": result["selected_epoch"] + 1,
                "best_val_score": result["best_val_macro_f1"],
                "best_val_macro_f1": result["best_val_macro_f1"],
                "test_primary_mean": result["overall_test_macro_f1"],
                "test_macro_f1_mean": result["overall_test_macro_f1"],
                "train_macro_f1": result["train_macro_f1"],
                "runtime_seconds": result["runtime_seconds"],
                "epochs_trained": result["epochs_ran"],
                "guard_fallback": result["guard_fallback"],
                "allocation_method": result["allocation_method"],
                "selected_loss_outer": result["selected_loss_outer"],
                "selected_loss_cls": result["selected_loss_cls"],
                "selected_environment_risk_variance": result["selected_environment_risk_variance"],
                "run_status": result.get("run_status", "COMPLETE"),
                **{f"test_{name}_macro_f1": value for name, value in result["group_f1"].items()},
            }
        )
        period_rows.extend(
            {"seed": result["seed"], "test_snapshot": idx, "macro_f1": value}
            for idx, value in result["period_f1"].items()
        )
        group_rows.extend(
            {"seed": result["seed"], "test_period": name, "macro_f1": value}
            for name, value in result["group_f1"].items()
        )
    write_csv(output_dir / "all_runs.csv", all_rows)
    write_csv(output_dir / "test_period_summary.csv", period_rows)
    write_csv(output_dir / "test_group_summary.csv", group_rows)

    lines = [
        "# TAFI Elliptic Results (SING Split)",
        "",
        "- Train snapshots: `6-10`",
        "- Validation snapshots: `11-15`",
        "- Test snapshots: `16-48`",
        "- Training mode: `snapshotwise` (fixed)",
        "- Metric: macro-F1",
        "- Primary inference rule: strict standard argmax",
        "",
        "## Test periods",
        "",
        "| Test period | Mean | Std. dev. |",
        "|---|---:|---:|",
    ]
    for name, _ in ELLIPTIC_TEST_GROUPS:
        values = [float(result["group_f1"][name]) for result in results]
        lines.append(f"| {name} | {mean(values):.6f} | {sample_std(values):.6f} |")
    overall = [float(result["overall_test_macro_f1"]) for result in results]
    lines.extend(
        [
            "",
            f"Overall: `{mean(overall):.6f} ± {sample_std(overall):.6f}`",
            "",
            "## Resolved hyperparameters",
            "",
            "```json",
            json.dumps(profile, indent=2, ensure_ascii=False, default=scalarize),
            "```",
            "",
        ]
    )
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"completed; report={output_dir / 'FINAL_REPORT.md'}", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)


    if args.datasets.strip().lower() == "all":
        selected_datasets = list(SUPPORTED_DATASETS)
    else:
        selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        for d in selected_datasets:
            if d not in SUPPORTED_DATASETS:
                raise ValueError(f"Unknown dataset {d!r}. Supported: {SUPPORTED_DATASETS}")

    good_shifts = [s.strip() for s in args.good_shifts.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    hyperparams = get_hyperparams(args)
    profile_override = get_profile_overrides(args)


    save_config(args, args.output_dir / "run_config_resolved.json")
    try:
        save_config(args, args.output_dir / "run_config_resolved.yaml")
    except Exception:
        pass

    print("=" * 70)
    print(" TAFI BENCHMARK RUNNER ")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Target datasets: {selected_datasets}")
    print(f"Random seeds: {seeds}")
    print(f"Output directory: {args.output_dir}")
    if profile_override:
        print(f"User profile overrides: {profile_override}")
    print("=" * 70)

    if "Elliptic" in selected_datasets:
        elliptic_output_dir = (
            args.output_dir
            if len(selected_datasets) == 1
            else args.output_dir / "Elliptic"
        )
        _run_elliptic(args, elliptic_output_dir)
        selected_datasets = [d for d in selected_datasets if d != "Elliptic"]
        if not selected_datasets:
            return


    print(f"Discovering settings under {args.data_root}...")
    settings = load_selected_settings(
        data_root=args.data_root,
        selected_datasets=selected_datasets,
        good_shifts=good_shifts,
        good_domain=args.good_domain,
        profile_override=profile_override,
    )

    if not settings:
        raise RuntimeError(f"No experiment settings found for {selected_datasets} under {args.data_root}")

    print(f"Successfully loaded {len(settings)} experimental setting(s):")
    for setting in settings:
        resolved = resolve_hyperparams(setting.profile, hyperparams)
        print(
            f"  - {setting.key} | metric={setting.main_metric} | epochs={setting.profile['epochs']} | "
            f"hidden={resolved['predictor_hidden_dim']} | layers={resolved['predictor_num_layers']} | "
            f"K={resolved['num_envs']} | rho={resolved['perturb_budget_fraction']} | "
            f"protect_eps={resolved['protect_eps']} | priority_floor={resolved['priority_floor']} | "
            f"inner_steps={resolved['inner_steps']} | tests={[name for name, _ in setting.tests]}"
        )

    total_runs = len(settings) * len(seeds)
    print(f"\nTotal planned runs: {total_runs}\n")

    all_rows: List[Dict[str, Any]] = []
    current_run = 0

    for setting in settings:
        for seed in seeds:
            current_run += 1
            print(f">>> [{current_run}/{total_runs}] Starting {setting.key} (seed {seed}) <<<")
            try:
                result = train_one_run(
                    setting=setting,
                    seed=seed,
                    device=device,
                    hyperparams=hyperparams,
                    output_dir=args.output_dir,
                    log_every=args.log_every,
                    save_checkpoint=not args.no_checkpoint,
                    save_predictions=args.save_predictions,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                result = _failure_result(
                    setting=setting,
                    seed=seed,
                    hyperparams=hyperparams,
                    output_dir=args.output_dir,
                    exc=exc,
                )
                print(
                    f"[SCIENTIFIC FAILURE {setting.key} | seed {seed}] "
                    f"{result['run_status']}: {result['failure_reason']}",
                    flush=True,
                )
            all_rows.append(result)
            write_csv(args.output_dir / "all_runs.partial.csv", all_rows)


    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_family.setdefault(r["family"], []).append(r)

    print("\n" + "=" * 70)
    print(" SAVING PER-DATASET ISOLATED RESULTS ")
    print("=" * 70)
    for fam, fam_rows in by_family.items():
        fam_dir = args.output_dir / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_csv = fam_dir / "all_runs.csv"
        fam_existing = load_csv_rows(fam_csv)
        fam_merged = merge_run_records(fam_existing, fam_rows)
        write_csv(fam_csv, fam_merged)
        fam_complete = [
            r for r in fam_merged
            if str(r.get("run_status", "COMPLETE")) == "COMPLETE"
            and r.get("test_accuracy_mean") not in (None, "")
        ]
        if fam_complete:
            fam_summary, fam_report = aggregate_results(fam_complete)
        else:
            fam_summary, fam_report = [], "No COMPLETE runs; see all_runs.csv for failure rows.\n"
        write_csv(fam_dir / "summary.csv", fam_summary)
        (fam_dir / "SUMMARY_REPORT.md").write_text(fam_report, encoding="utf-8")
        save_config(args, fam_dir / "run_config_resolved.json")
        print(f"  [{fam}] -> {fam_dir / 'summary.csv'} and {fam_dir / 'SUMMARY_REPORT.md'}")


    master_csv = args.output_dir / "all_runs.csv"
    master_existing = load_csv_rows(master_csv)
    master_merged = merge_run_records(master_existing, all_rows)
    write_csv(master_csv, master_merged)
    master_complete = [
        r for r in master_merged
        if str(r.get("run_status", "COMPLETE")) == "COMPLETE"
        and r.get("test_accuracy_mean") not in (None, "")
    ]
    if master_complete:
        master_summary, master_report = aggregate_results(master_complete)
    else:
        master_summary, master_report = [], "No COMPLETE runs; see all_runs.csv for failure rows.\n"
    write_csv(args.output_dir / "summary.csv", master_summary)
    (args.output_dir / "SUMMARY_REPORT.md").write_text(master_report, encoding="utf-8")

    partial_csv = args.output_dir / "all_runs.partial.csv"
    if partial_csv.is_file():
        partial_csv.unlink()

    print("\n" + "=" * 70)
    print(" RUN EXECUTION FINISHED; CHECK RUN STATUS ")
    print("=" * 70)
    print(f"Master results directory: {args.output_dir}")
    print(f"  - Master summary table: {args.output_dir / 'SUMMARY_REPORT.md'}")
    print(f"  - Master aggregated CSV: {args.output_dir / 'summary.csv'}")
    print(f"  - Master full runs CSV: {args.output_dir / 'all_runs.csv'}")
    print("=" * 70)
    print("\n" + master_report)


if __name__ == "__main__":
    main()
