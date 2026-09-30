"""Amendment 2 of protocol.md: zero-GPU late-fusion headroom on Video_Games."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import caption_signal as cs

GAMES_OFFSET = 66082  # downstream id = global id - 66082 + 1
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)


def win(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--florence-manifest", type=Path, required=True)
    parser.add_argument("--qwen-manifest", type=Path, required=True)
    parser.add_argument("--title-emb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    florence = {int(r["global_id"]): r for r in cs.load_verified_shards(args.florence_manifest, cs.FLORENCE_MANIFEST_SHA256)}
    remap, _join = cs.asin_join(args.pairs_dir, florence, allow_remap=True)
    split = {s: cs.load_pairs(args.pairs_dir, remap, s) for s in ("train", "valid", "test")}
    _c, domains, caps, titles = cs.build_domains(cs.load_verified_shards(args.qwen_manifest, None), "caption_raw")
    items = domains[cs.GAMES]
    item_set = set(items)
    title_tok = {i: cs.tokens(titles[i]) for i in items}
    raw_cap = {i: cs.tokens(caps[i]) for i in items}
    df = Counter(t for i in items for t in raw_cap[i])
    template = {t for t, c in df.items() if c / len(items) > cs.DF_MAX}
    cap_tok = {i: raw_cap[i] - template for i in items}
    doc_freq = Counter(t for i in items for t in (title_tok[i] | cap_tok[i]))
    idf = {t: float(np.log(len(items) / (1 + c)) + 1.0) for t, c in doc_freq.items()}
    donor = cs.derangement(sorted(items), random.Random(cs.SEED))

    emb = np.load(args.title_emb).astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    row = lambda g: g - GAMES_OFFSET + 1  # noqa: E731
    history = [(a, b) for a, b in split["train"] if a in item_set and b in item_set and a != b]
    popularity = Counter(b for _a, b in history)
    for i in items:
        popularity.setdefault(i, 0)

    def resid(a, x, caps_of):
        t = title_tok[a] | title_tok[x]
        return cs.wjaccard(caps_of(a) - t, caps_of(x) - t, idf)

    def triples(name, seed):
        pos = sorted({(a, b) for a, b in split[name] if a in item_set and b in item_set and a != b})
        negs = cs.matched_negatives(pos, items, popularity, {i: len(raw_cap[i]) for i in items},
                                    set(pos) | set(history), random.Random(seed))
        rows = []
        for p, ((a, b), ns) in enumerate(zip(pos, negs)):
            for c in ns:
                rows.append((p, emb[row(a)] @ emb[row(b)], emb[row(a)] @ emb[row(c)],
                             resid(a, b, cap_tok.__getitem__), resid(a, c, cap_tok.__getitem__),
                             resid(a, b, lambda i: cap_tok[donor[i]]), resid(a, c, lambda i: cap_tok[donor[i]])))
        return np.array(rows), len(pos)

    def per_positive(arr, n, lam, real):
        cp, cn = (3, 4) if real else (5, 6)
        w = win(arr[:, 1] + lam * arr[:, cp], arr[:, 2] + lam * arr[:, cn])
        num = np.bincount(arr[:, 0].astype(int), weights=w, minlength=n)
        den = np.bincount(arr[:, 0].astype(int), minlength=n).astype(float)
        return num, den

    auc = lambda num, den: float(num.sum() / den.sum())  # noqa: E731
    valid, nv = triples("valid", cs.SEED + 1)
    test, nt = triples("test", cs.SEED + 2)
    choose = {real: max(LAMBDAS, key=lambda l: auc(*per_positive(valid, nv, l, real))) for real in (True, False)}

    base_n, base_d = per_positive(test, nt, 0.0, True)
    real_n, _ = per_positive(test, nt, choose[True], True)
    shuf_n, _ = per_positive(test, nt, choose[False], False)
    rng = np.random.default_rng(cs.SEED)
    idx = rng.integers(0, nt, size=(cs.BOOT, nt))
    keep = base_d > 0

    def boot(num):
        return (num[idx] * keep[idx]).sum(1) / (base_d[idx] * keep[idx]).sum(1)

    b_base, b_real, b_shuf = boot(base_n), boot(real_n), boot(shuf_n)
    ci = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]  # noqa: E731
    gain_real = auc(real_n, base_d) - auc(base_n, base_d)
    gain_shuf = auc(shuf_n, base_d) - auc(base_n, base_d)
    out = {
        "valid_positives": nv, "test_positives": nt, "lambda_real": choose[True], "lambda_shuffle": choose[False],
        "valid_auc_by_lambda_real": {str(l): auc(*per_positive(valid, nv, l, True)) for l in LAMBDAS},
        "test_auc_title": auc(base_n, base_d), "test_auc_fused_real": auc(real_n, base_d),
        "test_auc_fused_shuffle": auc(shuf_n, base_d),
        "gain_real": gain_real, "gain_real_ci95": ci(b_real - b_base),
        "gain_shuffle": gain_shuf, "gain_shuffle_ci95": ci(b_shuf - b_base),
        "real_minus_shuffle_ci95": ci(b_real - b_shuf),
    }
    out["pass"] = bool(gain_real >= 0.005 and out["gain_real_ci95"][0] > 0 and out["real_minus_shuffle_ci95"][0] > 0)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
