"""Zero-GPU caption screening; implements ``protocol.md`` exactly.

Measures, per domain, whether residual caption tokens (caption minus both
titles) separate the true next item from popularity- and caption-length-
matched negatives when the titles cannot (title-tied triples).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an the of on in with and or to for is are be its it this that these those by at from as into "
    "has have there their them which who what featuring features shown shows showing image picture photo "
    "product item set pack pcs piece pieces".split()
)
DF_MAX = 0.3
K = 5
SEED = 20260930
BOOT = 1000
FLORENCE_MANIFEST_SHA256 = "521849b297ae9dfce3626f60db48c0facbeaa757bfc343bff2285d3d89af84d7"
GAMES = "Video_Games"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in TOKEN.findall(text.lower()) if len(t) >= 2 and t not in STOP)


def load_verified_shards(manifest_path: Path, expected_sha: str | None) -> list[dict]:
    if expected_sha is not None and sha256_file(manifest_path) != expected_sha:
        raise RuntimeError(f"manifest hash mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completion_status") != "complete":
        raise RuntimeError(f"incomplete manifest: {manifest_path}")
    records = []
    for shard in manifest["shards"]:
        path = manifest_path.parent / shard["path"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {path}")
        records += [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    return records


def load_pairs(pairs_dir: Path, remap: dict[int, int] | None, split: str = "train") -> list[tuple[int, int]]:
    pairs = []
    with gzip.open(pairs_dir / f"pairs_{split}.tsv.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            _user, a, b, _n = line.rstrip("\n").split("\t")
            a, b = int(a), int(b)
            pairs.append((remap[a], remap[b]) if remap is not None else (a, b))
    return pairs


def asin_join(pairs_dir: Path, catalog: dict[int, dict], allow_remap: bool) -> tuple[dict[int, int] | None, dict]:
    """Map CSV item ids to catalog global ids by ASIN; for AmazonMix-6 the map must be the identity."""
    by_asin = {str(r["asin"]): gid for gid, r in catalog.items() if r.get("asin")}
    remap, mismatched, missing = {}, 0, 0
    with gzip.open(pairs_dir / "item_asin.tsv.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            item_id, asin = line.rstrip("\n").split("\t")
            gid = by_asin.get(asin)
            if gid is None:
                missing += 1
                continue
            remap[int(item_id)] = gid
            mismatched += gid != int(item_id)
    report = {"csv_items": len(remap) + missing, "asin_missing_in_catalog": missing, "id_differs_from_global_id": mismatched}
    if missing:
        raise RuntimeError(f"ASIN join failed: {report}")
    if not allow_remap:
        if mismatched:
            raise RuntimeError(f"CSV item ids are not catalog global ids: {report}")
        return None, report
    return remap, report


def matched_negatives(positives, items, popularity, cap_len, positive_set, rng):
    """Same popularity quintile and caption-length tertile as the positive target."""
    ranked = sorted(items, key=lambda i: (popularity[i], i))
    pop_bin = {item: min(4, 5 * rank // len(ranked)) for rank, item in enumerate(ranked)}
    by_len = sorted(items, key=lambda i: (cap_len[i], i))
    len_bin = {item: min(2, 3 * rank // len(by_len)) for rank, item in enumerate(by_len)}
    groups = defaultdict(list)
    for item in items:
        groups[(pop_bin[item], len_bin[item])].append(item)
    out = []
    for a, b in positives:
        pool = groups[(pop_bin[b], len_bin[b])]
        chosen, tries = [], 0
        while len(chosen) < K and tries < 20 * K:
            tries += 1
            c = pool[rng.randrange(len(pool))]
            if c in (a, b) or c in chosen or (a, c) in positive_set:
                continue
            chosen.append(c)
        out.append(chosen)
    return out


def derangement(items: list[int], rng: random.Random) -> dict[int, int]:
    shuffled = items[:]
    while True:
        rng.shuffle(shuffled)
        if all(x != y for x, y in zip(items, shuffled)):
            return dict(zip(items, shuffled))


def wjaccard(x: frozenset, y: frozenset, idf: dict[str, float]) -> float:
    inter = x & y
    if not inter:
        return 0.0
    return sum(idf[t] for t in inter) / sum(idf[t] for t in x | y)


def auc_and_ci(num: np.ndarray, den: np.ndarray, rng: np.random.Generator) -> tuple[float, list[float], np.ndarray]:
    keep = den > 0
    num, den = num[keep], den[keep]
    if not keep.any():
        raise RuntimeError("empty AUC subset")
    point = float(num.sum() / den.sum())
    idx = rng.integers(0, len(num), size=(BOOT, len(num)))
    boots = num[idx].sum(1) / den[idx].sum(1)
    return point, [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], boots


def score_domain(name, items, captions, titles, pairs, seed_offset, train_pairs=None):
    """Primary M: ``pairs`` are train transitions. Amendment 1 (M_cf): ``pairs`` are
    test transitions and ``train_pairs`` supply popularity and the CF tie condition."""
    rng = random.Random(SEED + seed_offset)
    item_set = set(items)
    title_tok = {i: tokens(titles[i]) for i in items}
    raw_cap = {i: tokens(captions[i]) for i in items}
    df = Counter(t for i in items for t in raw_cap[i])
    template = {t for t, c in df.items() if c / len(items) > DF_MAX}
    cap_tok = {i: raw_cap[i] - template for i in items}
    doc_freq = Counter(t for i in items for t in (title_tok[i] | cap_tok[i]))
    idf = {t: math.log(len(items) / (1 + c)) + 1.0 for t, c in doc_freq.items()}

    in_domain = [(a, b) for a, b in pairs if a in item_set and b in item_set and a != b]
    positives = sorted(set(in_domain))
    positive_set = set(positives)
    history = in_domain
    cf = None
    if train_pairs is not None:
        history = [(a, b) for a, b in train_pairs if a in item_set and b in item_set and a != b]
        positive_set |= set(history)
        cf = Counter((min(a, b), max(a, b)) for a, b in history)
    popularity = Counter(b for _a, b in history)
    for i in items:
        popularity.setdefault(i, 0)
    negatives = matched_negatives(positives, items, popularity, {i: len(raw_cap[i]) for i in items}, positive_set, rng)
    donor = derangement(sorted(items), rng)

    def resid(a, x, caps):
        t = title_tok[a] | title_tok[x]
        return wjaccard(caps(a) - t, caps(x) - t, idf)

    real = cap_tok.__getitem__
    shuffled = lambda i: cap_tok[donor[i]]  # noqa: E731
    n = len(positives)
    keys = ["title", "cap_raw", "cap_resid", "main", "shuffle_main"] + (["main_cf", "shuffle_main_cf"] if cf is not None else [])
    acc = {key: (np.zeros(n), np.zeros(n)) for key in keys}
    ties_main = triples = tied = cf_tied = 0
    for p, ((a, b), negs) in enumerate(zip(positives, negatives)):
        st_pos = wjaccard(title_tok[a], title_tok[b], idf)
        raw_pos = wjaccard(cap_tok[a], cap_tok[b], idf)
        res_pos, shuf_pos = resid(a, b, real), resid(a, b, shuffled)
        for c in negs:
            triples += 1
            st_neg = wjaccard(title_tok[a], title_tok[c], idf)
            comps = {
                "title": (st_pos, st_neg),
                "cap_raw": (raw_pos, wjaccard(cap_tok[a], cap_tok[c], idf)),
                "cap_resid": (res_pos, resid(a, c, real)),
            }
            if st_pos == st_neg:
                tied += 1
                comps["main"] = comps["cap_resid"]
                comps["shuffle_main"] = (shuf_pos, resid(a, c, shuffled))
                ties_main += comps["main"][0] == comps["main"][1]
                if cf is not None and cf[(min(a, b), max(a, b))] == cf[(min(a, c), max(a, c))]:
                    cf_tied += 1
                    comps["main_cf"] = comps["main"]
                    comps["shuffle_main_cf"] = comps["shuffle_main"]
            for key, (x, y) in comps.items():
                acc[key][0][p] += 1.0 if x > y else 0.5 if x == y else 0.0
                acc[key][1][p] += 1.0

    brng = np.random.default_rng(SEED + seed_offset)
    result = {"mode": "primary_train" if cf is None else "amendment1_test_cf", "items_with_caption": len(items),
              "template_tokens_dropped": len(template), "positives": n, "triples": triples,
              "title_tied_share": tied / max(triples, 1), "main_tie_share": ties_main / max(tied, 1)}
    if cf is not None:
        result["title_and_cf_tied_share"] = cf_tied / max(triples, 1)
    boots = {}
    for key, (num, den) in acc.items():
        point, ci, boots[key] = auc_and_ci(num, den, brng)
        result[f"auc_{key}"] = point
        result[f"auc_{key}_ci95"] = ci
    return result, boots


def build_domains(records, caption_field, ok_status="ok"):
    catalog, domains = {}, defaultdict(list)
    captions, titles = {}, {}
    for r in records:
        gid = int(r["global_id"])
        catalog[gid] = r
        titles[gid] = str(r.get("title") or "")
        if r.get("caption_status") == ok_status and r.get(caption_field):
            captions[gid] = str(r[caption_field])
            domains[str(r["domain"])].append(gid)
    return catalog, {d: sorted(v) for d, v in domains.items()}, captions, titles


def decide(results: dict, boots: dict, metric: str, extra_gates: dict | None = None) -> dict:
    """Pre-registered rule applied to ``metric`` ("main" = M, "main_cf" = M_cf)."""
    shuffle = f"shuffle_{metric}"
    gates = {
        "G1_shuffle_null": all(0.49 <= r[f"auc_{shuffle}"] <= 0.51 and r[f"auc_{shuffle}_ci95"][0] <= 0.5 <= r[f"auc_{shuffle}_ci95"][1]
                              for r in results.values()),
        "G2_title_positive_control": all(r["auc_title"] >= 0.55 for r in results.values()),
    }
    ranking = sorted(results, key=lambda d: results[d][f"auc_{metric}"], reverse=True)
    gates["G3_games_not_top2"] = GAMES not in ranking[:2]
    gates.update(extra_gates or {})
    decision = {"metric": metric, "gates": gates, "ranking": ranking, "qualifying": None}
    diffs = {}
    for d in ranking:
        if d == GAMES:
            continue
        delta = boots[d][metric] - boots[GAMES][metric]
        diffs[d] = {"minus_games": results[d][f"auc_{metric}"] - results[GAMES][f"auc_{metric}"],
                    "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))]}
    decision["difference_vs_games"] = diffs
    if all(gates.values()):
        top = ranking[0]
        if top != GAMES and results[top][f"auc_{metric}"] - 0.5 >= 0.02 and diffs[top]["ci95"][0] > 0:
            decision["qualifying"] = top
    decision["verdict"] = ("PROXY_INVALID" if not all(gates.values())
                           else f"RUN_PAIRED_{decision['qualifying']}" if decision["qualifying"] else "CLOSE_CAPTION_DIRECTION")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--florence-manifest", type=Path, required=True)
    parser.add_argument("--qwen-games-manifest", type=Path)
    parser.add_argument("--allow-remap", action="store_true", help="fixture only: CSV ids are not global ids")
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    florence = load_verified_shards(args.florence_manifest, FLORENCE_MANIFEST_SHA256)
    catalog, domains, captions, titles = build_domains(florence, "caption_raw")
    remap, join = asin_join(args.pairs_dir, catalog, args.allow_remap)
    train = load_pairs(args.pairs_dir, remap, "train")
    test = load_pairs(args.pairs_dir, remap, "test")
    wanted = args.domains or sorted(domains)
    primary, amendment, p_boots, a_boots = {}, {}, {}, {}
    for offset, name in enumerate(sorted(domains)):
        if name not in wanted:
            continue
        primary[name], p_boots[name] = score_domain(name, domains[name], captions, titles, train, offset)
        amendment[name], a_boots[name] = score_domain(name, domains[name], captions, titles, test, offset, train)
        print(name, "M", round(primary[name]["auc_main"], 4), "M_cf", round(amendment[name]["auc_main_cf"], 4), flush=True)
    out = {"protocol": "research/caption_augmentation/screening/protocol.md", "caption_source": "Florence-2 v10",
           "asin_join": join, "train_pairs": len(train), "test_pairs": len(test),
           "primary_M": primary, "amendment1_M_cf": amendment}
    extra = {}
    if args.qwen_games_manifest:
        qwen = load_verified_shards(args.qwen_games_manifest, None)
        _c, q_domains, q_caps, q_titles = build_domains(qwen, "caption_raw")
        q_primary, _ = score_domain(GAMES, q_domains[GAMES], q_caps, q_titles, train, 99)
        q_amend, _ = score_domain(GAMES, q_domains[GAMES], q_caps, q_titles, test, 99, train)
        out["calibration_games_qwen3vl"] = {"primary_M": q_primary, "amendment1_M_cf": q_amend}
        extra["C_cf_qwen_games_Mcf_le_0.52"] = q_amend["auc_main_cf"] <= 0.52
    if GAMES in primary and len(primary) > 1:
        out["decision_primary_M"] = decide(primary, p_boots, "main")
        out["decision_amendment1_M_cf"] = decide(amendment, a_boots, "main_cf", extra)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("decision", "calibration"))}, indent=1)[:4000])


if __name__ == "__main__":
    main()
