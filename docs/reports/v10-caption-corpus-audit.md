# Phase 2 Corpus Audit — Final V10

## Verdict

**PASS for corpus integrity and arm assembly.** The final Kaggle V10 artifact contains the complete 108,753-row AmazonMix-6 catalog. The production loader re-read all shards, verified every manifest SHA-256, and reconstructed contiguous `global_id` order `0..108,752`.

This is an engineering/data-contract result only. It is not evidence that captions improve recommendation quality. Downstream CSFT, MNTP, SimCSE, extraction, SASRec, and independent rank audits remain unexecuted.

## Source artifact

- Kernel: `trixuanle/llm2rec-caption-full-corpus-v2-inline`
- Stage: `full_corpus_generation`
- Completion: `COMPLETE`
- Local audit input: downloaded Kaggle V10 `full_corpus` output (not committed; 213 shards)
- Shard manifest SHA-256: `521849b297ae9dfce3626f60db48c0facbeaa757bfc343bff2285d3d89af84d7`
- Caption revision: `f0acedbf9b780e04fe1f9111fcf53187388f3d03`
- Paraphraser: `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Shard size/count: `512` / `213`

## Coverage

| Quantity | Count | Rate |
|---|---:|---:|
| Catalog rows | 108,753 | 100.000% |
| Decoded images | 108,226 | 99.515% |
| Valid captions | 108,226 | 99.515% |
| Valid paraphrases | 108,226 | 99.515% |
| Rows retained without image/caption | 527 | 0.485% |

Image status breakdown for the 527 unavailable rows: `missing_metadata=409`, `download_failed=98`, `no_asin_in_catalog=20`. These rows remain in the corpus and are not silently dropped.

Catalog block counts match the registered six-domain contract:

- `Arts_Crafts_and_Sewing`: 12,454
- `Electronics`: 20,150
- `Home_and_Kitchen`: 33,478
- `Video_Games`: 9,517
- `Movies_and_TV`: 13,190
- `Tools_and_Home_Improvement`: 19,964

## Arm-contract checks

Five downstream arms were materialized from the validated records in ascending `global_id` order: `title-only`, `null`, `real`, `shuffle`, and `paraphrase`.

- Arm key set exact on all `108,753` rows.
- `title-only` equals the original title on all `108,753` rows.
- All cue-bearing arms preserve the exact `Title: <original title>` prefix on all `108,753` rows.
- The available-item set contains `108,226` IDs.
- Shuffle donors form a bijection over exactly that set: `108,226` unique donors, zero fixed points.
- No arm rows were dropped during assembly.

The temporary local arm materialization used `write_arm_corpora()` and produced one JSON map plus one line-oriented text file per arm. These files are derived artifacts, not committed binary outputs; the final Kaggle shard records remain the source of truth for downstream packaging.

## Interaction-weighted denominator currently available

The local Games train split provides a downstream sanity denominator only:

- 122,577 training interactions
- 8,488 unique target IDs
- 8,482 unique target IDs with valid captions
- unique-target coverage: 99.929%
- interaction-weighted coverage: 99.916%

This is not an all-domain interaction-weighted estimate. The remaining domains require their corresponding mixed-corpus interaction files or a Kaggle-side audit before reporting a global interaction-weighted denominator.

## Reproduction

The audit used the production loader and arm serializer, not a second parser:

```bash
cd research && python -m unittest discover -s caption_augmentation -p 'test_*.py'
```

The next execution boundary is the tiny real-data end-to-end chain. It must first consume this manifest, reject stale/wrong-arm inputs by hash, and prove target preservation, item order, masks, causal/bidirectional checks, and checkpoint ancestry before any full training matrix is scheduled.
