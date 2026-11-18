"""Disk cache for stage outputs.

Each stage records a key = hash(video fingerprint, relevant config sections,
upstream stage keys). If the key on disk matches, the stage is skipped. Changing
render settings therefore never re-runs the models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def video_fingerprint(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha1()
    h.update(str(p.stat().st_size).encode())
    with p.open("rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


class StageCache:
    def __init__(self, root: str | Path, fingerprint: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fp = fingerprint
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, str] = (
            json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        )

    def key(self, stage: str, *parts: str) -> str:
        return hashlib.sha1("|".join((self.fp, stage) + parts).encode()).hexdigest()[:16]

    def fresh(self, stage: str, key: str, *files: str) -> bool:
        return self.manifest.get(stage) == key and all((self.root / f).exists() for f in files)

    def commit(self, stage: str, key: str) -> None:
        self.manifest[stage] = key
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def path(self, name: str) -> Path:
        return self.root / name

    # convenience I/O
    def save_df(self, name: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.path(name), index=False)

    def load_df(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(name))

    def save_json(self, name: str, obj: Any) -> None:
        self.path(name).write_text(json.dumps(obj, indent=2, default=_np_default))

    def load_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text())

    def save_npz(self, name: str, **arrays) -> None:
        np.savez_compressed(self.path(name), **arrays)

    def load_npz(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.path(name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
