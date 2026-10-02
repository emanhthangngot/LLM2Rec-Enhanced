---
phase: 3
title: "Matched follow-up and investment decision"
status: waiting-external
priority: P1
---

# Phase 3: Matched follow-up and investment decision

## Dependency and scope

Only proceed after the new scientifically eligible SFT parent is COMPLETE, hashes match, and clean/noisy diagnostic evidence is read. A failed reference gate blocks follow-up. No test scoring, recipe rescue, or unlimited seed matrix.

## Selected next experiment

The completed probe shows strong transient perturbation but cannot establish training harm. First execute matched semantic-only NoDO-on/off: same source/parent/data/cohort/seed, w1.0, one epoch; scientific variable sigma0.05 vs0.0 only. Prospective protocol `reports/nodo-ablation-protocol.json`; implementation/verification plan `reports/nodo-ablation-plan.md`. New notebook `kaggle/hanorec-v11-nodo-ablation/notebook.ipynb` cap9000s (2.5h) leaves7200s conservatively unreserved for a subsequent CF decision if justified.

The generic four-control CF notebook remains prepared but is NOT submitted blindly. Sigma0 is an ablation, not a replacement HaNoRec baseline. No ten-core-source file is changed; per-arm execution configurations and hashed receipts distinguish the two same-weight sigma variants.

Submission: `trlxun/hanorec-v11-nodo-ablation` v1 accepted through Kaggle MCP, accounta/trlxun, T4, timeout9000s. Completion is not observed. Pre-push evidence is in `reports/nodo-ablation-verification.md`; cumulative worst-case reservations are21600s, leaving7200s unreserved. Exact billed GPU time remains unknown. No automatic polling; wait for the user's completion/result notice before downloading and auditing this run.

## Verification and decisions

- Hard MCP follow-up timeout <=16200 seconds (4.5 session GPU-hours); cumulative additional spend including failures <=28800 seconds (8 hours).
- Read actual diagnostics before choosing the exact follow-up. If noise/objective is broken, replan within the remaining ceiling instead of running the prepared matrix blindly.
- Full-cohort metrics use 265 sampled events; user-cluster uncertainty must report actual unique users and conditional fitted-model scope.
- This ablation has one primary off-minus-on NDCG contrast. Recall and each-minus-SFT are exploratory. The registered two-contrast CF family remains separate and unexecuted; do not conflate the two experiment families.
- A continue-investment claim also requires credible gain beyond SFT-only at the predeclared +0.002 NDCG threshold; point estimates alone cannot pass.
- Whole-user sign-flip inference assumes within-user label exchangeability; bootstrap does not estimate training-seed uncertainty.
- Preserve historical negative/inconclusive evidence and publish all planned comparisons, including losses.

## Checklist

- [ ] Execute the chosen matched follow-up within the remaining budget.
- [ ] Conclude continue, stop-investment, or inconclusive with explicit scientific limits.
