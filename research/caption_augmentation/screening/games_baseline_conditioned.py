"""Video_Games only: does the Qwen3-VL caption carry next-item signal on the
test transitions that the trained title-only LLM2Rec embedding gets wrong, and
does the trained `real`-arm embedding recover it?

Uses the exact pilot test split and the two trained embeddings; negatives
follow protocol.md (popularity quintile x caption-length tertile, K=5).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import caption_signal as cs

GAMES_OFFSET = 66082  # downstream id = global id - 66082 + 1 (dataset contract)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, required=True, help="Games 5-core pairs (CSV id space)")
    parser.add_argument("--florence-manifest", type=Path, required=True)
    parser.add_argument("--qwen-manifest", type=Path, required=True)
    parser.add_argument("--title-emb", type=Path, required=True)
    parser.add_argument("--real-emb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    florence = {int(r["global_id"]): r for r in cs.load_verified_shards(args.florence_manifest, cs.FLORENCE_MANIFEST_SHA256)}
    remap, join = cs.asin_join(args.pairs_dir, florence, allow_remap=True)
    train = cs.load_pairs(args.pairs_dir, remap, "train")
    test = cs.load_pairs(args.pairs_dir, remap, "test")
    _c, domains, caps, titles = cs.build_domains(cs.load_verified_shards(args.qwen_manifest, None), "caption_raw")
    items = domains[cs.GAMES]
    item_set = set(items)

    title_tok = {i: cs.tokens(titles[i]) for i in items}
    raw_cap = {i: cs.tokens(caps[i]) for i in items}
    df = Counter(t for i in items for t in raw_cap[i])
    template = {t for t, c in df.items() if c / len(items) > cs.DF_MAX}
    cap_tok = {i: raw_cap[i] - template for i in items}
    doc_freq = Counter(t for i in items for t in (title_tok[i] | cap_tok[i]))
    idf = {t: np.log(len(items) / (1 + c)) + 1.0 for t, c in doc_freq.items()}

    history = [(a, b) for a, b in train if a in item_set and b in item_set and a != b]
    positives = sorted({(a, b) for a, b in test if a in item_set and b in item_set and a != b})
    popularity = Counter(b for _a, b in history)
    for i in items:
        popularity.setdefault(i, 0)
    negatives = cs.matched_negatives(positives, items, popularity, {i: len(raw_cap[i]) for i in items},
                                     set(positives) | set(history), random.Random(cs.SEED))

    def unit(path):
        m = np.load(path).astype(np.float64)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    emb_t, emb_r = unit(args.title_emb), unit(args.real_emb)
    row = lambda g: g - GAMES_OFFSET + 1  # noqa: E731

    def resid(a, x):
        t = title_tok[a] | title_tok[x]
        return cs.wjaccard(cap_tok[a] - t, cap_tok[x] - t, idf)

    def win(x, y):
        return 1.0 if x > y else 0.5 if x == y else 0.0

    buckets = {"all": [], "title_emb_wrong": [], "title_emb_right": []}
    for (a, b), negs in zip(positives, negatives):
        ea_t, ea_r = emb_t[row(a)], emb_r[row(a)]
        for c in negs:
            t_pos, t_neg = ea_t @ emb_t[row(b)], ea_t @ emb_t[row(c)]
            rec = (win(resid(a, b), resid(a, c)), win(t_pos, t_neg), win(ea_r @ emb_r[row(b)], ea_r @ emb_r[row(c)]))
            buckets["all"].append(rec)
            buckets["title_emb_wrong" if t_pos <= t_neg else "title_emb_right"].append(rec)

    out = {"asin_join": join, "positives": len(positives), "negatives_per_positive": cs.K}
    for name, recs in buckets.items():
        arr = np.array(recs)
        out[name] = {"triples": len(arr), "auc_caption_resid": float(arr[:, 0].mean()),
                     "auc_title_only_embedding": float(arr[:, 1].mean()), "auc_real_embedding": float(arr[:, 2].mean())}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
