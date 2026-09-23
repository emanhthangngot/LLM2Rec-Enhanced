"""One-step real Qwen2 CSFT smoke for title-only and caption arms.

This is deliberately a technical execution smoke. It proves that the verified
V10 arm text reaches a real causal-LM optimizer with original-title targets; it
does not produce recommendation metrics or a reusable scientific checkpoint.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

CATALOG_SIZE = 108_753
GAMES_START = 66_082
GAMES_STOP = 75_599
ARMS = ("title-only", "real")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def install_runtime() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "transformers==4.44.2"],
        check=True,
    )


def find_input_file(name: str) -> Path:
    matches = sorted(path for path in Path("/kaggle/input").rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"input file not found: {name}")
    return matches[-1]


def load_corpus() -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = find_input_file("v10-corpus-manifest.json")
    root = manifest_path.parent
    transport = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source = "521849b297ae9dfce3626f60db48c0facbeaa757bfc343bff2285d3d89af84d7"
    if transport.get("source_manifest_sha256") != expected_source:
        raise RuntimeError("V10 source manifest identity mismatch")
    names = [str(transport["record_file"]), str(transport["record_file"]).removesuffix(".gz")]
    matches = sorted(
        path for name in names for path in Path("/kaggle/input").rglob(name) if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"V10 record transport not found: {names}")
    record_path = matches[-1]
    payload = record_path.read_bytes()
    raw = gzip.decompress(payload) if record_path.name.endswith(".gz") else payload
    expected_hash = transport["record_file_sha256_uncompressed"]
    if sha256_bytes(raw) != expected_hash:
        raise RuntimeError("V10 uncompressed transport hash mismatch")
    records = [json.loads(line.decode("utf-8")) for line in raw.split(b"\n") if line]
    if len(records) != CATALOG_SIZE or [r.get("global_id") for r in records] != list(range(CATALOG_SIZE)):
        raise RuntimeError("V10 record count/order contract failed")
    return transport, records


def choose_examples(records: list[dict[str, object]], count: int = 4) -> list[dict[str, object]]:
    train_files = sorted(
        path for path in Path("/kaggle/input").rglob("*.csv")
        if path.is_file() and "train" in str(path).lower()
    )
    if not train_files:
        raise FileNotFoundError("Games train CSV not found")
    examples: list[dict[str, object]] = []
    with train_files[-1].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                history_local = [int(x) for x in ast.literal_eval(row["history_item_id"])]
                target_local = int(row["item_id"])
            except (KeyError, SyntaxError, ValueError, TypeError):
                continue
            history_global = [GAMES_START + item for item in history_local]
            target_global = GAMES_START + target_local
            if not history_global or not (GAMES_START <= target_global < GAMES_STOP):
                continue
            if any(records[item].get("caption_status") != "ok" for item in history_global):
                continue
            if records[target_global]["title"] != row["item_title"]:
                raise RuntimeError(f"target title mismatch for local item {target_local}")
            examples.append({
                "user_id": row["user_id"],
                "history_global_ids": history_global[-3:],
                "target_global_id": target_global,
                "target_title": records[target_global]["title"],
            })
            if len(examples) == count:
                return examples
    raise RuntimeError(f"only found {len(examples)} valid Games examples; need {count}")


def padded_batch(tokenizer, examples: list[tuple[str, str]], max_length: int):
    import torch
    rows = []
    for prompt, target in examples:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
        rows.append((ids, labels))
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, (ids, row_labels) in enumerate(rows):
        input_ids[index, :len(ids)] = torch.tensor(ids)
        labels[index, :len(row_labels)] = torch.tensor(row_labels)
        mask[index, :len(ids)] = 1
    return input_ids, mask, labels


def run_arm(arm: str, examples: list[dict[str, object]], records: list[dict[str, object]], tokenizer, model_id: str, revision: str, max_length: int) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM
    prompts = []
    for example in examples:
        history_text = " ".join(records[item]["arm_texts"][arm] for item in example["history_global_ids"])
        prompts.append((history_text, example["target_title"]))
    input_ids, attention_mask, labels = padded_batch(tokenizer, prompts, max_length)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    parameter = next(model.parameters())
    before = float(parameter.detach().flatten()[0].item())
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), labels=labels.cuda()
    )
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite CSFT loss for arm {arm}: {loss.item()}")
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - started
    after = float(parameter.detach().flatten()[0].item())
    if not torch.isfinite(parameter.detach()).all():
        raise RuntimeError(f"optimizer produced non-finite parameters for arm {arm}")
    model.eval()
    with torch.no_grad():
        post_output = model(
            input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), labels=labels.cuda()
        )
        post_loss = post_output.loss
    if not torch.isfinite(post_loss):
        raise RuntimeError(f"post-step loss is non-finite for arm {arm}: {post_loss.item()}")
    if before == after:
        raise RuntimeError(f"optimizer did not change model parameters for arm {arm}")
    result = {
        "arm": arm,
        "optimizer_steps": 1,
        "loss_after_step": float(post_loss.detach().cpu().item()),
        "loss_before_step": float(loss.detach().cpu().item()),
        "parameter_probe_before": before,
        "parameter_probe_after": after,
        "parameter_changed": True,
        "elapsed_seconds": elapsed,
        "target_global_ids": [e["target_global_id"] for e in examples],
        "history_global_ids": [e["history_global_ids"] for e in examples],
        "input_hash": sha256_bytes(json.dumps(prompts, ensure_ascii=False, sort_keys=True).encode()),
        "checkpoint_ancestry": {"parent": None, "completed_step": 1},
    }
    del optimizer, model, output, post_output, loss, post_loss
    torch.cuda.empty_cache()
    return result

def main() -> None:
    protocol = {
        "corpus_dataset": "trixuanle/llm2rec-caption-corpus-v10-compact",
        "development_dataset": "trixuanle/llm2rec-games-5core",
        "rows_per_arm": 4,
        "max_length": 128,
        "model_id": "Qwen/Qwen2-0.5B",
        "model_revision": "91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
        "tokenizer_revision": "91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
        "optimizer_steps": 1,
        "training_scope": "technical smoke, not recommendation efficacy",
    }
    install_runtime()
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("tiny real CSFT requires a CUDA GPU")
    transport, records = load_corpus()
    examples = choose_examples(records, int(protocol["rows_per_arm"]))
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["model_id"], revision=protocol["tokenizer_revision"], use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    results = [
        run_arm(arm, examples, records, tokenizer, protocol["model_id"], protocol["model_revision"], protocol["max_length"])
        for arm in ARMS
    ]
    artifact = {
        "status": "PASS",
        "stage": "tiny_real_csft_chain",
        "training_executed": True,
        "technical_smoke_only": True,
        "corpus_source_manifest_sha256": transport["source_manifest_sha256"],
        "record_count": len(records),
        "protocol": protocol,
        "examples": examples,
        "arms": results,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    Path("/kaggle/working/tiny-csft-artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
