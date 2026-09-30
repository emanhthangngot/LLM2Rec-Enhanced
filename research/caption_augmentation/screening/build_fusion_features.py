"""Amendment 3: build frozen item-feature matrices for the late-fusion SASRec run.

Rows follow the downstream item ids (row 0 = null item), matching the pilot
title-only embedding. Writes ``title.npy`` (pilot copy), ``fused_real.npy``,
``fused_shuffle.npy`` and ``features_manifest.json`` with SHA-256 values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import caption_signal as cs

GAMES_OFFSET = 66082
CATALOG = 9517
DIM = 256
MIN_DF = 3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-manifest", type=Path, required=True)
    parser.add_argument("--title-emb", type=Path, required=True)
    parser.add_argument("--title-emb-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if sha(args.title_emb) != args.title_emb_sha256:
        raise RuntimeError("pilot title-only embedding hash mismatch")
    title_emb = np.load(args.title_emb)
    if title_emb.shape != (CATALOG + 1, 896) or not np.isfinite(title_emb).all():
        raise RuntimeError(f"unexpected title embedding {title_emb.shape}")

    _c, domains, caps, titles = cs.build_domains(cs.load_verified_shards(args.qwen_manifest, None), "caption_raw")
    records = {int(r["global_id"]): r for r in cs.load_verified_shards(args.qwen_manifest, None)}
    if sorted(records) != list(range(GAMES_OFFSET, GAMES_OFFSET + CATALOG)):
        raise RuntimeError("Qwen corpus does not cover the Video_Games catalog exactly")
    items = domains[cs.GAMES]
    raw_cap = {i: cs.tokens(caps[i]) for i in items}
    df = Counter(t for i in items for t in raw_cap[i])
    template = {t for t, c in df.items() if c / len(items) > cs.DF_MAX}
    resid = {i: raw_cap[i] - template - cs.tokens(titles[i]) for i in items}
    vocab_df = Counter(t for i in items for t in resid[i])
    vocab = sorted(t for t, c in vocab_df.items() if c >= MIN_DF)
    col = {t: j for j, t in enumerate(vocab)}
    idf = np.array([np.log(len(items) / (1 + vocab_df[t])) + 1.0 for t in vocab])

    X = np.zeros((CATALOG + 1, len(vocab)), dtype=np.float64)
    for gid in items:
        cols = [col[t] for t in resid[gid] if t in col]
        if cols:
            X[gid - GAMES_OFFSET + 1, cols] = idf[cols]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    # Unsupervised SVD on item features only (no interactions).
    _u, _s, vt = np.linalg.svd(X[1:], full_matrices=False)
    E = X @ vt[:DIM].T
    en = np.linalg.norm(E, axis=1, keepdims=True)
    E = np.divide(E, en, out=np.zeros_like(E), where=en > 0)
    scale = float(np.linalg.norm(title_emb[1:], axis=1).mean())

    donor = cs.derangement(list(range(1, CATALOG + 1)), random.Random(cs.SEED))
    E_shuf = E.copy()
    E_shuf[1:] = E[[donor[i] for i in range(1, CATALOG + 1)]]

    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "title.npy": title_emb.astype(np.float32),
        "fused_real.npy": np.hstack([title_emb, scale * E]).astype(np.float32),
        "fused_shuffle.npy": np.hstack([title_emb, scale * E_shuf]).astype(np.float32),
    }
    manifest = {"vocab_size": len(vocab), "svd_dim": DIM, "min_df": MIN_DF, "scale": scale,
                "items_with_caption_feature": int((np.linalg.norm(E[1:], axis=1) > 0).sum()),
                "title_emb_sha256": args.title_emb_sha256, "files": {}}
    for name, matrix in outputs.items():
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"non-finite values in {name}")
        np.save(args.out / name, matrix)
        manifest["files"][name] = {"shape": list(matrix.shape), "sha256": sha(args.out / name)}
    (args.out / "features_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
