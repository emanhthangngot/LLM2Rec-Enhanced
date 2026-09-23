# Phase 3 Paired Bootstrap Audit

## Contract

- **Outcome:** test the pre-registered real-image versus shuffled-image gate using the six completed arm artifacts.
- **Constraints:** local CPU-only analysis; paired users and targets; no new GPU run; preserve the fixed candidate protocol.
- **Non-goals:** no new seed, no model retraining, no metric or candidate-set changes, no statistical claim beyond this selected 265-user slice.
- **Acceptance:** verify paired keys, compute per-user deltas for NDCG@10 and Recall@10, and report deterministic 95% paired-bootstrap intervals.

## Method

Source files: the six downloaded `arm_result.json` files under
`research/hanorec_cf_hardness/results/full_run_265/hanorec-cf-hardness-arm-w{10,00,05}-{real,shuffle}/`.

Each real arm was paired with its shuffle control by `(user_id, target)`. All
six arms had exactly 265 rows; all paired key sets and targets matched. For
each metric and weight, the audit computed the per-user difference
`real - shuffle`, then generated 20,000 bootstrap resamples of the 265 paired
users using deterministic seed `20260920`. The interval is the percentile 2.5%
to 97.5% interval of the bootstrap mean difference.

## Results

| Weight | Metric | Mean delta | Median delta | 95% bootstrap CI | Positive-user fraction | Gate |
|---:|---|---:|---:|---|---:|---|
| 1.0 | NDCG@10 | 0.0080569 | 0.0000000 | [-0.0018450, 0.0180222] | 5.28% | Not met |
| 1.0 | Recall@10 | 0.0226415 | 0.0000000 | [-0.0037736, 0.0490566] | 3.40% | Not met |
| 0.0 | NDCG@10 | 0.0092724 | 0.0000000 | [-0.0012437, 0.0199733] | 5.28% | Not met |
| 0.0 | Recall@10 | 0.0264151 | 0.0000000 | [0.0000000, 0.0528302] | 3.77% | Not met under strict `CI > 0` rule |
| 0.5 | NDCG@10 | 0.0070011 | 0.0000000 | [-0.0029529, 0.0171577] | 4.91% | Not met |
| 0.5 | Recall@10 | 0.0188679 | 0.0000000 | [-0.0037736, 0.0452830] | 3.02% | Not met |

The positive point estimates are driven by a small number of users whose
ranked target changes. Most paired users have zero delta; this is why the
median delta is zero for every cell and why the intervals are wide relative to
the means.

## Decision

The pre-registered gate is not passed by any cell. The result remains a
**descriptive real-image advantage**, not a confirmed effect. In particular,
`w=0.0` has the best point estimate among real-image arms, but its advantage is
not sufficient to promote CF-only hardness as a validated winner.

No additional GPU training is justified by this gate alone. A future repeated-
seed study is a separate decision and should be opened only with an explicit
new contract and budget.
