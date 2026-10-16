"""Twitch-Explicit data loader with MUSAE node indexing."""


from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp


TWITCH_LANGS = ("DE", "ENGB", "ES", "FR", "PTBR", "RU", "TW")
TWITCH_FEATURE_DIM = 3170


def _resolve_twitch_language_dir(data_dir: str | Path, lang: str) -> Path:

    root = Path(data_dir).expanduser().resolve()
    direct = root / lang
    nested = root / "twitch" / lang

    if direct.is_dir():
        return direct
    if nested.is_dir():
        return nested


    return direct


def load_twitch(data_dir: str | Path, lang: str) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:

    if lang not in TWITCH_LANGS:
        raise AssertionError(f"Invalid Twitch language {lang!r}; expected one of {TWITCH_LANGS}")

    folder = _resolve_twitch_language_dir(data_dir, lang)
    target_path = folder / f"musae_{lang}_target.csv"
    edges_path = folder / f"musae_{lang}_edges.csv"
    features_path = folder / f"musae_{lang}_features.json"

    missing = [p for p in (target_path, edges_path, features_path) if not p.is_file()]
    if missing:
        expected = "\n".join(str(p) for p in (target_path, edges_path, features_path))
        raise FileNotFoundError(
            f"Missing Twitch {lang} file(s): {', '.join(str(p) for p in missing)}\n"
            f"Expected SING/MUSAE layout:\n{expected}"
        )

    labels = []
    node_ids = []
    seen_ids = set()

    # Zero-based CSV indices: node ID at row[5], mature label at row[2].
    with target_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for line_no, row in enumerate(reader, start=2):
            if len(row) <= 5:
                raise ValueError(
                    f"{target_path}: row {line_no} has {len(row)} columns; SING expects >= 6"
                )
            node_id = int(row[5])
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            labels.append(1 if row[2].strip() == "True" else 0)
            node_ids.append(node_id)

    node_ids_arr = np.asarray(node_ids, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    n = int(labels_arr.shape[0])
    if n == 0:
        raise ValueError(f"{target_path}: no target rows were loaded")

    src = []
    dst = []
    with edges_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for line_no, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"{edges_path}: row {line_no} has fewer than 2 columns")
            src.append(int(row[0]))
            dst.append(int(row[1]))

    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    if src_arr.size and (src_arr.min() < 0 or dst_arr.min() < 0 or src_arr.max() >= n or dst_arr.max() >= n):
        raise ValueError(
            f"{edges_path}: edge node ids must lie in [0, {n - 1}] to match SING indexing"
        )

    adjacency = sp.csr_matrix(
        (np.ones(src_arr.shape[0], dtype=np.float32), (src_arr, dst_arr)),
        shape=(n, n),
    )

    with features_path.open("r", encoding="utf-8") as handle:
        feature_map = json.load(handle)

    features = np.zeros((n, TWITCH_FEATURE_DIM), dtype=np.float32)
    for raw_node, raw_features in feature_map.items():
        node = int(raw_node)

        if node < 0 or node >= n:
            continue
        cols = np.asarray(raw_features, dtype=np.int64)
        if cols.size == 0:
            continue
        if cols.min() < 0 or cols.max() >= TWITCH_FEATURE_DIM:
            raise ValueError(
                f"{features_path}: node {node} contains feature index outside "
                f"[0, {TWITCH_FEATURE_DIM - 1}]"
            )
        features[node, cols] = 1.0

    # The MUSAE target CSV is not guaranteed to be row-sorted by graph node id.
    # SING explicitly restores label[i] to graph node i using the node-id column.
    inverse = {node_id: row_idx for row_idx, node_id in enumerate(node_ids_arr.tolist())}
    missing_native_ids = [i for i in range(n) if i not in inverse]
    if missing_native_ids:
        preview = missing_native_ids[:10]
        raise ValueError(
            f"{target_path}: missing native node ids required by SING reordering: {preview}"
            + (" ..." if len(missing_native_ids) > 10 else "")
        )
    reorder = np.fromiter((inverse[i] for i in range(n)), dtype=np.int64, count=n)
    labels_arr = labels_arr[reorder]

    return adjacency, labels_arr, features


__all__ = ["TWITCH_LANGS", "TWITCH_FEATURE_DIM", "load_twitch"]
