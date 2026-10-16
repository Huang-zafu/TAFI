"""Dataset loading utilities for Twitch-Explicit and OGB-Arxiv."""


from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from load_data import TWITCH_LANGS, load_twitch


class NCDataset:
    def __init__(self, name: str, root: str | Path = "./data") -> None:
        self.name = name
        self.root = str(root)
        self.graph: Dict[str, object] = {}
        self.label = None

    def __getitem__(self, idx: int):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"


def load_nc_dataset(
    data_dir: str | Path,
    dataname: str,
    sub_dataname: str = "",
    year: Sequence[int] | int = 2020,
):

    if dataname == "twitch-e":
        if sub_dataname not in TWITCH_LANGS:
            raise ValueError(
                f"Invalid Twitch language {sub_dataname!r}; expected one of {TWITCH_LANGS}"
            )
        return load_twitch_dataset(data_dir, sub_dataname)

    if dataname == "ogb-arxiv":
        if isinstance(year, (int, np.integer)):
            year_bound = [2018, int(year)]
        else:
            year_bound = [int(year[0]), int(year[1])]
        return load_ogb_arxiv(data_dir=data_dir, year_bound=year_bound, proportion=1.0)

    raise ValueError(f"Invalid dataname: {dataname!r}")


def load_twitch_dataset(data_dir: str | Path, lang: str) -> NCDataset:

    if lang not in TWITCH_LANGS:
        raise AssertionError(f"Invalid Twitch language {lang!r}")

    adjacency, label, features = load_twitch(data_dir, lang)
    rows, cols = adjacency.nonzero()
    edge_index = torch.as_tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    node_feat = torch.as_tensor(features, dtype=torch.float32)

    dataset = NCDataset(lang, root=data_dir)
    dataset.graph = {
        "edge_index": edge_index,
        "edge_feat": None,
        "node_feat": node_feat,
        "num_nodes": int(node_feat.shape[0]),
    }
    dataset.label = torch.as_tensor(label, dtype=torch.long)
    return dataset


_OGB_RAW_CACHE: Dict[str, Tuple[dict, np.ndarray]] = {}


def _resolve_ogb_root(data_dir: str | Path) -> Path:

    path = Path(data_dir).expanduser().resolve()
    normalized = path.name.lower().replace("-", "_")
    return path.parent if normalized == "ogbn_arxiv" else path


def _load_ogb_raw(data_dir: str | Path) -> Tuple[dict, np.ndarray]:
    root = _resolve_ogb_root(data_dir)
    cache_key = str(root)
    cached = _OGB_RAW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from ogb.nodeproppred import NodePropPredDataset
    except Exception as exc:
        raise ImportError(
            "ogb is required for ogbn_arxiv. Install it in the same Python environment "
            "with: python -m pip install ogb"
        ) from exc

    raw = NodePropPredDataset(name="ogbn-arxiv", root=str(root))
    graph = raw.graph
    labels = np.asarray(raw.labels)
    _OGB_RAW_CACHE[cache_key] = (graph, labels)
    return graph, labels


def _build_ogb_year_subgraph(
    graph: dict,
    labels: np.ndarray,
    year_bound: Sequence[int],
    proportion: float,
) -> NCDataset:

    if len(year_bound) != 2:
        raise ValueError("year_bound must contain [lower, upper]")
    lower, upper = int(year_bound[0]), int(year_bound[1])
    if lower > upper:
        raise ValueError(f"invalid year_bound {year_bound}: lower > upper")
    if not 0.0 < float(proportion) <= 1.0:
        raise ValueError("proportion must lie in (0, 1]")

    node_year = np.asarray(graph["node_year"]).reshape(-1)
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    node_feat = np.asarray(graph["node_feat"])
    n = int(node_year.shape[0])

    # SING computes degree using only edges whose two endpoints are no newer than
    # the split's upper year, then ranks all eligible nodes by that degree.
    eligible_edge = (node_year[edge_index[0]] <= upper) & (node_year[edge_index[1]] <= upper)
    selected_edges = edge_index[:, eligible_edge]
    degree = np.zeros(n, dtype=np.int64)
    np.add.at(degree, selected_edges[0], 1)
    np.add.at(degree, selected_edges[1], 1)

    eligible_nodes = np.flatnonzero(node_year <= upper)
    # Stable descending sort reproduces Python's stable sort used by SING for ties.
    order = np.argsort(-degree[eligible_nodes], kind="stable")
    ranked_nodes = eligible_nodes[order]
    keep_count = int(float(proportion) * ranked_nodes.shape[0])
    ranked_nodes = ranked_nodes[:keep_count]

    remap = np.full(n, -1, dtype=np.int64)
    remap[ranked_nodes] = np.arange(ranked_nodes.shape[0], dtype=np.int64)
    edge_keep = (remap[edge_index[0]] >= 0) & (remap[edge_index[1]] >= 0)
    induced = edge_index[:, edge_keep]
    remapped_edges = np.stack([remap[induced[0]], remap[induced[1]]], axis=0)

    result_features = node_feat[ranked_nodes]
    result_labels = np.asarray(labels)[ranked_nodes]
    result_years = node_year[ranked_nodes]

    dataset = NCDataset("ogb-arxiv")
    dataset.graph = {
        "edge_index": torch.as_tensor(remapped_edges, dtype=torch.long),
        "edge_feat": None,
        "node_feat": torch.as_tensor(result_features, dtype=torch.float32),
        "num_nodes": int(ranked_nodes.shape[0]),
    }
    dataset.label = torch.as_tensor(result_labels, dtype=torch.long)
    dataset.test_mask = torch.as_tensor(result_years > lower, dtype=torch.bool)
    dataset.year_bound = (lower, upper)
    return dataset


def load_ogb_arxiv(
    data_dir: str | Path,
    year_bound: Sequence[int] = (2018, 2020),
    proportion: float = 1.0,
) -> NCDataset:
    graph, labels = _load_ogb_raw(data_dir)
    return _build_ogb_year_subgraph(graph, labels, year_bound, proportion)


__all__ = [
    "NCDataset",
    "load_nc_dataset",
    "load_twitch_dataset",
    "load_ogb_arxiv",
]
