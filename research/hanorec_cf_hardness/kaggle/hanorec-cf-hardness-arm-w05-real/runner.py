"""Real HaNoRec x LLM2Rec CF-hardness pipeline -- resumable per-arm script
runner. Plan: plans/260918-1114-hanorec-cf-hardness/plan.md.
Kernel type: script (not notebook) -- avoids the confirmed nbclient/Papermill
1800s cell-reply timeout that cancelled a prior long-running notebook push in
this repo (plans/260915-0955-visual-delta-fusion-pilot/ISSUES.md#22)."""
from __future__ import annotations

import subprocess, sys

PINS = ["transformers==4.51.3", "peft==0.15.2", "accelerate==1.6.0", "bitsandbytes==0.45.5"]
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *PINS], check=True)
import torch
print({"cuda": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
if not torch.cuda.is_available():
    raise RuntimeError("This kernel requires exactly one visible GPU; none detected.")


import json
import math
import time
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Verbatim-ported HaRS math (hanorec/hars/math.py, commit 587face7).
# --------------------------------------------------------------------------

def _finite(values, name):
    checked = [float(v) for v in values]
    if not checked:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(v) for v in checked):
        raise ValueError(f"{name} must contain only finite values")
    return checked


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def hars_softmax(values):
    checked = _finite(values, "values")
    offset = max(checked)
    weights = [math.exp(v - offset) for v in checked]
    total = sum(weights)
    return [w / total for w in weights]


def hars_probability_distance(chosen, rejected):
    chosen_v = _finite(chosen, "chosen")
    rejected_v = _finite(rejected, "rejected")
    if len(chosen_v) != len(rejected_v):
        raise ValueError("chosen and rejected must have equal length")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(chosen_v, rejected_v)))


def hars_normalize_hardness(deltas):
    checked = _finite(deltas, "deltas")
    if any(d < 0 for d in checked):
        raise ValueError("deltas must be non-negative")
    denominator = _stable_sigmoid(sum(checked) / len(checked))
    return [_stable_sigmoid(d) / denominator for d in checked]


def cf_normalize_hardness(margins):
    """Frozen Phase 1 formula: lambda_cf = sigmoid(m) / sigmoid(mean(m)).

    Unlike hars_normalize_hardness, margins are signed (a retriever score
    difference), so this is intentionally not clamped to (0, 1); see
    plans/260918-1114-hanorec-cf-hardness/phase-01-local-spec-and-audit.md#2.
    """
    checked = _finite(margins, "margins")
    denominator = _stable_sigmoid(sum(checked) / len(checked))
    return [_stable_sigmoid(m) / denominator for m in checked]


def responsiveness(reward_gaps):
    """Torch-free port of hanorec/hars/math.py::model_responsiveness (Eq. 8)."""
    checked = _finite(reward_gaps, "reward_gaps")
    eps = 1e-8
    mean_gap = sum(checked) / len(checked)
    if abs(mean_gap) > eps:
        scale = mean_gap
    else:
        scale = max(sum(abs(v) for v in checked) / len(checked), eps)
    normalized = [v / scale for v in checked]
    trimmed = sorted(normalized)[1:-1] if len(normalized) > 2 else normalized
    trimmed_mean = sum(trimmed) / len(trimmed)
    normalized_mean = sum(normalized) / len(normalized)
    return _stable_sigmoid(trimmed_mean) / _stable_sigmoid(normalized_mean)


# --------------------------------------------------------------------------
# Prompt construction shared by SFT, DPO, and reranking.
# --------------------------------------------------------------------------

_QUESTION = "Based on the user's history, will they like this candidate item next? Answer Yes or No."


def _resize_image(image_path: str, max_pixels: int):
    from PIL import Image

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return image


def _build_messages(catalog: dict[str, dict[str, str]], history: list[int], candidate: int, max_pixels: int, shuffle_map: dict[int, int] | None):
    def item_image(item_id: int):
        source_id = item_id if shuffle_map is None else shuffle_map.get(item_id, item_id)
        return _resize_image(catalog[str(source_id)]["image_path"], max_pixels)

    content: list[dict[str, Any]] = [{"type": "text", "text": "User history:"}]
    for history_item in history:
        content.append({"type": "image", "image": item_image(history_item)})
        content.append({"type": "text", "text": catalog[str(history_item)]["title"]})
    content.append({"type": "text", "text": "Candidate item:"})
    content.append({"type": "image", "image": item_image(candidate)})
    content.append({"type": "text", "text": catalog[str(candidate)]["title"]})
    content.append({"type": "text", "text": _QUESTION})
    return [{"role": "user", "content": content}]


def _semantic_embedding(model, processor, torch_module, title: str, image):
    tokenizer = processor.tokenizer
    inputs = tokenizer(title, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    with torch_module.no_grad():
        text_states = model.model.embed_tokens(inputs["input_ids"].to(device))
        mask = inputs["attention_mask"].to(device).unsqueeze(-1).to(text_states.dtype)
        text_vector = (text_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        image_inputs = processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device)
        grid_thw = image_inputs["image_grid_thw"].to(device)
        visual_dtype = next(model.visual.parameters()).dtype
        pixel_values = pixel_values.to(visual_dtype)
        visual_features = model.visual(pixel_values, grid_thw)
        visual_vector = visual_features.mean(dim=0, keepdim=True).to(text_vector.dtype)

    text_np = text_vector[0].float().cpu().numpy()
    visual_np = visual_vector[0].float().cpu().numpy()
    return text_np, visual_np


def _fuse_and_topk(text_matrix, visual_matrix, item_order: list[int], k: int):
    import numpy as np

    def normalize_rows(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

    fused = normalize_rows(normalize_rows(text_matrix) + normalize_rows(visual_matrix))
    similarity = fused @ fused.T
    neighbors: dict[int, dict[str, list[float]]] = {}
    for row_index, item_id in enumerate(item_order):
        row = similarity[row_index].copy()
        row[row_index] = -np.inf
        top_k = min(k, len(item_order) - 1)
        selected = np.argpartition(-row, top_k - 1)[:top_k]
        selected = sorted(selected.tolist(), key=lambda index: (-float(row[index]), item_order[index]))
        neighbors[item_id] = {
            "item_ids": [item_order[index] for index in selected],
            "scores": [float(row[index]) for index in selected],
        }
    return neighbors


# --------------------------------------------------------------------------
# NoDO: real transient LoRA perturbation hooks (hanorec/nodo/hooks.py port).
# --------------------------------------------------------------------------

def _iter_active_lora_pairs(model):
    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        active = getattr(module, "active_adapters", None) or getattr(module, "active_adapter", None)
        if isinstance(active, str):
            active = [active]
        for adapter_name in active or []:
            if adapter_name in lora_a and adapter_name in lora_b:
                yield module_name, adapter_name, lora_a[adapter_name], lora_b[adapter_name]


class _NodoPerturbation:
    def __init__(self, model, sigma: float, torch_module, functional_module):
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self._torch = torch_module
        self._functional = functional_module
        self._pairs = list(_iter_active_lora_pairs(model))
        if not self._pairs:
            raise RuntimeError("NoDO requires at least one active LoRA A/B module pair")
        self._handles: list[Any] = []

    def __enter__(self):
        torch_module = self._torch

        def make_hook(noise):
            def hook(_module, inputs, output):
                return output + self._functional.linear(inputs[0], noise, None)

            return hook

        for _module_name, _adapter_name, lora_a, lora_b in self._pairs:
            noise_a = torch_module.randn_like(lora_a.weight) * float(self._sigma)
            noise_b = torch_module.randn_like(lora_b.weight) * float(self._sigma)
            self._handles.append(lora_a.register_forward_hook(make_hook(noise_a)))
            self._handles.append(lora_b.register_forward_hook(make_hook(noise_b)))
        return self

    def __exit__(self, *exc_info):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles = []
        return False


def perturb_lora_weights(model, sigma: float, torch_module, functional_module):
    perturbation = _NodoPerturbation(model, sigma, torch_module, functional_module)
    perturbation._sigma = sigma
    return perturbation


# --------------------------------------------------------------------------
# Shared model-call helpers (module-level, not closures, so both the
# single-push train_and_evaluate() and the resumable run_sft()/run_arm()
# split below can reuse identical logic without duplicating it).
# --------------------------------------------------------------------------

def _build_inputs(processor, policy_model, catalog, history: list[int], candidate: int, max_pixels: int, shuffle_map: dict[int, int] | None):
    messages = _build_messages(catalog, history, candidate, max_pixels, shuffle_map)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [content["image"] for content in messages[0]["content"] if content["type"] == "image"]
    return processor(text=[text], images=images, return_tensors="pt").to(policy_model.device)


def _answer_logprobs(model, model_inputs, functional_module, yes_token_id: int, no_token_id: int) -> tuple:
    """One forward pass; returns (yes_logprob, no_logprob) predicted at the
    position right after the prompt. Reading both tokens from the same
    forward avoids a redundant second pass (previously one forward per
    token checked)."""
    outputs = model(**model_inputs)
    logits = outputs.logits[:, -1, :]
    log_probs = functional_module.log_softmax(logits.float(), dim=-1)
    return log_probs[0, yes_token_id], log_probs[0, no_token_id]


def _reference_answer_logprobs(policy_model, model_inputs, torch_module, functional_module, yes_token_id: int, no_token_id: int) -> tuple:
    """Real frozen-reference forward pass via PEFT's disable_adapter().

    get_peft_model() wraps the loaded model in place (shared tensors), so
    there is no independent second copy of the pretrained weights. The
    correct, standard way to get a deterministic pretrained-only forward
    pass from the same object is to disable the LoRA adapter and force
    eval mode (so dropout is inactive) for the duration of the call, then
    restore train mode so subsequent policy forwards behave as trained.
    """
    was_training = policy_model.training
    policy_model.eval()
    try:
        with torch_module.no_grad(), policy_model.disable_adapter():
            values = _answer_logprobs(policy_model, model_inputs, functional_module, yes_token_id, no_token_id)
    finally:
        if was_training:
            policy_model.train()
    return values


def _build_dpo_examples(train_pairs, yes_token_id: int, no_token_id: int, shuffle_map: dict[int, int] | None):
    examples = []
    for pair in train_pairs:
        examples.append(
            {
                "history": pair["history"],
                "candidate": pair["positive"],
                "chosen_token": yes_token_id,
                "rejected_token": no_token_id,
                "shuffle_map": shuffle_map,
            }
        )
        examples.append(
            {
                "history": pair["history"],
                "candidate": pair["negative"],
                "chosen_token": no_token_id,
                "rejected_token": yes_token_id,
                "shuffle_map": shuffle_map,
            }
        )
    return examples


def _load_base_model(config: dict[str, Any]):
    """Real 4-bit Qwen2.5-VL load, pinned to a single GPU (device_map={"": 0})
    -- Kaggle's 2xT4 auto-sharding caused real OOMs when left as "auto".
    Returns the RAW (not LoRA-wrapped) model: semantic embedding extraction
    (_semantic_embedding) walks `model.model.embed_tokens`/`model.visual`,
    which only exist on the unwrapped `Qwen2_5_VLForConditionalGeneration`
    object -- get_peft_model() wraps it in a PeftModel container that does
    not expose those attributes at the same path. Wrap with _wrap_lora()
    only after any embedding extraction is done."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    device_map = {"": 0}
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    policy_model.eval()
    return processor, policy_model


def _wrap_lora(policy_model):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(policy_model, lora_config)


def _run_dpo_arm(
    config: dict[str, Any],
    processor,
    policy_model,
    catalog,
    train_pairs,
    evaluation_rows,
    lambda_sem_values,
    lambda_cf_values,
    item_order,
    weight: float,
    image_condition: str,
    yes_token_id: int,
    no_token_id: int,
    sft_state: dict,
    remaining,
    log_stage,
):
    """Real DPO training + reranking for exactly one (weight, image_condition)
    arm. Shared by the single-push train_and_evaluate() loop and the
    standalone run_arm() entry point -- identical math either way."""
    import math as math_module
    import numpy as np
    import torch
    import torch.nn.functional as functional

    max_pixels = int(config.get("max_pixels", 50176))
    dpo_steps = int(config.get("dpo_steps", 2))
    noise_sigma = float(config.get("noise_sigma", 0.05))
    beta0 = float(config.get("beta0", 0.1))

    for name, value in policy_model.named_parameters():
        if value.requires_grad and name in sft_state:
            value.data.copy_(sft_state[name])
    pre_training_snapshot = {name: value.detach().clone() for name, value in policy_model.named_parameters() if value.requires_grad}

    lambda_combined = [
        lambda_sem_values[i] ** weight * lambda_cf_values[i] ** (1.0 - weight) for i in range(len(train_pairs))
    ]
    if math_module.isclose(weight, 1.0):
        for computed, expected in zip(lambda_combined, lambda_sem_values):
            if not math_module.isclose(computed, expected, rel_tol=1e-9):
                raise RuntimeError("w=1.0 must reduce lambda_combined to lambda_sem exactly")
    if math_module.isclose(weight, 0.0):
        for computed, expected in zip(lambda_combined, lambda_cf_values):
            if not math_module.isclose(computed, expected, rel_tol=1e-9):
                raise RuntimeError("w=0.0 must reduce lambda_combined to lambda_cf exactly")

    rng = np.random.default_rng(int(config.get("seed", 2024)))
    permutation = item_order.copy()
    rng.shuffle(permutation)
    shuffle_map = dict(zip(item_order, permutation))
    arm_shuffle_map = None if image_condition == "real" else shuffle_map

    examples = _build_dpo_examples(train_pairs, yes_token_id, no_token_id, arm_shuffle_map)
    hardness_per_example = [value for value in lambda_combined for _ in range(2)]

    optimizer = torch.optim.AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=float(config["learning_rate"]))
    policy_model.train()

    step_losses: list[float] = []
    arm_stats_seconds = 0.0
    arm_train_seconds = 0.0
    for step in range(dpo_steps):
        if remaining() <= 0:
            break
        stats_pass_start = time.monotonic()
        stats_logits: list[float] = []
        for example in examples:
            if remaining() <= 0:
                break
            chosen_inputs = _build_inputs(processor, policy_model, catalog, example["history"], example["candidate"], max_pixels, example["shuffle_map"])
            with torch.no_grad():
                policy_yes, policy_no = _answer_logprobs(policy_model, chosen_inputs, functional, yes_token_id, no_token_id)
            reference_yes, reference_no = _reference_answer_logprobs(policy_model, chosen_inputs, torch, functional, yes_token_id, no_token_id)
            is_chosen_yes = example["chosen_token"] == yes_token_id
            policy_chosen = policy_yes if is_chosen_yes else policy_no
            policy_rejected = policy_no if is_chosen_yes else policy_yes
            reference_chosen = reference_yes if is_chosen_yes else reference_no
            reference_rejected = reference_no if is_chosen_yes else reference_yes
            stats_logits.append(float((policy_chosen - reference_chosen - policy_rejected + reference_rejected).cpu()))
        arm_stats_seconds += time.monotonic() - stats_pass_start
        reward_gaps = [beta0 * value for value in stats_logits]
        if len(reward_gaps) < 3:
            raise ValueError("HaNoRec Eq. 8 requires at least 3 examples in the global mini-batch")
        scale = responsiveness(reward_gaps)
        betas = [max(1e-6, beta0 * scale * hardness_per_example[i]) for i in range(len(examples))]

        train_pass_start = time.monotonic()
        optimizer.zero_grad()
        per_example_losses: list[float] = []
        for index, example in enumerate(examples):
            if remaining() <= 0:
                break
            chosen_inputs = _build_inputs(processor, policy_model, catalog, example["history"], example["candidate"], max_pixels, example["shuffle_map"])
            with perturb_lora_weights(policy_model, noise_sigma, torch, functional):
                policy_yes, policy_no = _answer_logprobs(policy_model, chosen_inputs, functional, yes_token_id, no_token_id)
            reference_yes, reference_no = _reference_answer_logprobs(policy_model, chosen_inputs, torch, functional, yes_token_id, no_token_id)
            is_chosen_yes = example["chosen_token"] == yes_token_id
            policy_chosen = policy_yes if is_chosen_yes else policy_no
            policy_rejected = policy_no if is_chosen_yes else policy_yes
            reference_chosen = reference_yes if is_chosen_yes else reference_no
            reference_rejected = reference_no if is_chosen_yes else reference_yes
            preference_logit = policy_chosen - reference_chosen - policy_rejected + reference_rejected
            example_loss = -functional.logsigmoid(betas[index] * preference_logit) / len(examples)
            if not torch.isfinite(example_loss):
                raise RuntimeError("HaNoRec DPO loss produced a non-finite value")
            example_loss.backward()
            per_example_losses.append(float(example_loss.detach().cpu()))
        optimizer.step()
        step_losses.append(sum(per_example_losses))
        arm_train_seconds += time.monotonic() - train_pass_start
    log_stage(f"dpo_stats_pass_w{weight}_{image_condition}", arm_stats_seconds, units=len(step_losses) * len(examples))
    log_stage(f"dpo_train_pass_w{weight}_{image_condition}", arm_train_seconds, units=len(step_losses) * len(examples))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    changed = any(
        not torch.equal(value, pre_training_snapshot[name])
        for name, value in policy_model.named_parameters()
        if value.requires_grad
    )
    if not changed and step_losses:
        raise RuntimeError("no LoRA parameter changed after DPO training; NoDO/backprop path is broken")

    predictions = []
    policy_model.eval()
    rerank_start = time.monotonic()
    total_candidates_scored = 0
    with torch.no_grad():
        for eval_row in evaluation_rows:
            if remaining() <= 0:
                break
            candidate_scores = []
            for candidate in eval_row["candidates"]:
                candidate_inputs = _build_inputs(processor, policy_model, catalog, eval_row["history"], candidate, max_pixels, arm_shuffle_map)
                yes_logprob, no_logprob = _answer_logprobs(policy_model, candidate_inputs, functional, yes_token_id, no_token_id)
                candidate_scores.append(float((yes_logprob - no_logprob).cpu()))
                total_candidates_scored += 1
            ranked = sorted(zip(eval_row["candidates"], candidate_scores), key=lambda pair: -pair[1])
            ranked_ids = [item for item, _ in ranked]
            target = eval_row["target"]
            rank = ranked_ids.index(target) + 1 if target in ranked_ids else 0
            ndcg10 = (1.0 / math_module.log2(rank + 1)) if 0 < rank <= 10 else 0.0
            recall10 = 1.0 if 0 < rank <= 10 else 0.0
            candidate_recall20 = 1.0 if target in eval_row["candidates"] else 0.0
            predictions.append(
                {
                    "user_id": eval_row["user_id"],
                    "target": target,
                    "ranked_candidates": ranked_ids,
                    "rank": rank,
                    "ndcg@10": ndcg10,
                    "recall@10": recall10,
                    "candidate_recall@20": candidate_recall20,
                }
            )
    log_stage(f"rerank_w{weight}_{image_condition}", time.monotonic() - rerank_start, units=total_candidates_scored)
    policy_model.train()

    return {
        "weight": weight,
        "image_condition": image_condition,
        "status": "COMPLETE" if step_losses else "SKIPPED_BUDGET",
        "losses": step_losses,
        "predictions": predictions,
        "mean_ndcg@10": sum(p["ndcg@10"] for p in predictions) / len(predictions) if predictions else None,
        "mean_recall@10": sum(p["recall@10"] for p in predictions) / len(predictions) if predictions else None,
        "mean_candidate_recall@20": (
            sum(p["candidate_recall@20"] for p in predictions) / len(predictions) if predictions else None
        ),
        "arm_state": {name: value.detach().clone() for name, value in policy_model.named_parameters() if value.requires_grad},
    }


def _compute_semantic_hardness(policy_model, processor, catalog, train_pairs, max_pixels: int, log_stage):
    """Real semantic HaRS Top-K over the train-referenced catalog subset.
    Returns (lambda_sem_values, lambda_cf_values, item_order)."""
    import numpy as np
    import torch

    embedding_start = time.monotonic()
    item_order = sorted(int(item_id) for item_id in catalog.keys())
    top_k = min(10, len(catalog) - 1)
    text_vectors = []
    visual_vectors = []
    for item_id in item_order:
        record = catalog[str(item_id)]
        image = _resize_image(record["image_path"], max_pixels)
        text_vector, visual_vector = _semantic_embedding(policy_model, processor, torch, record["title"], image)
        text_vectors.append(text_vector)
        visual_vectors.append(visual_vector)
    log_stage("semantic_embedding", time.monotonic() - embedding_start, units=len(item_order))

    neighbors = _fuse_and_topk(np.stack(text_vectors), np.stack(visual_vectors), item_order, top_k)

    def pair_semantic_delta(positive_id: int, negative_id: int) -> float:
        positive_scores = neighbors[positive_id]["scores"]
        negative_scores = neighbors[negative_id]["scores"]
        return hars_probability_distance(hars_softmax(positive_scores), hars_softmax(negative_scores))

    semantic_deltas = [pair_semantic_delta(pair["positive"], pair["negative"]) for pair in train_pairs]
    lambda_sem_values = hars_normalize_hardness(semantic_deltas)
    cf_margins = [pair["cf_margin"] for pair in train_pairs]
    lambda_cf_values = cf_normalize_hardness(cf_margins)
    return lambda_sem_values, lambda_cf_values, item_order


# --------------------------------------------------------------------------
# Resumable split: run_sft() (one push) + run_arm() (one push per arm),
# for real multi-session execution when a single push cannot finish all 6
# arms inside one real Kaggle session (see
# plans/260918-1114-hanorec-cf-hardness/phase-02-full-dataset-scaling.md#7).
# --------------------------------------------------------------------------

def run_sft(config: dict[str, Any], prepared: dict[str, Any], output_root: Path, hanorec_root: Path) -> dict[str, Any]:
    """Real model load + semantic hardness + SFT only. Persists sft_lora_state.pt
    and a portable sft_bundle.json (train pairs, eval rows, catalog with
    output_root-relative image paths, lambda_sem/lambda_cf, item_order) so a
    later run_arm() push -- in a fresh kernel mounting this one's output via
    kernel_sources -- can resume without redownloading images or rescoring."""
    import torch
    import torch.nn.functional as functional

    output_root.mkdir(parents=True, exist_ok=True)
    stage_timings: list[dict[str, Any]] = []

    def log_stage(name: str, seconds: float, units: int | None = None) -> None:
        payload: dict[str, Any] = {"stage": name, "seconds": seconds}
        if units:
            payload["units"] = units
            payload["seconds_per_unit"] = seconds / units
        stage_timings.append(payload)
        print(json.dumps({"timing": payload}))

    start_time = time.monotonic()
    deadline = start_time + float(config.get("budget_seconds", 6600))

    def remaining() -> float:
        return deadline - time.monotonic()

    catalog = prepared["catalog"]
    train_pairs = prepared["train"]
    evaluation_rows = prepared["evaluation"]
    max_pixels = int(config.get("max_pixels", 50176))

    model_load_start = time.monotonic()
    processor, policy_model = _load_base_model(config)
    log_stage("model_load", time.monotonic() - model_load_start)

    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    lambda_sem_values, lambda_cf_values, item_order = _compute_semantic_hardness(
        policy_model, processor, catalog, train_pairs, max_pixels, log_stage
    )

    policy_model = _wrap_lora(policy_model)
    policy_model.train()
    optimizer = torch.optim.AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=float(config["learning_rate"]))

    sft_start = time.monotonic()
    sft_steps = int(config.get("sft_steps", 2))
    sft_losses: list[float] = []
    for step in range(sft_steps):
        if remaining() <= 0:
            break
        optimizer.zero_grad()
        per_example_sft_losses: list[float] = []
        for pair in train_pairs:
            model_inputs = _build_inputs(processor, policy_model, catalog, pair["history"], pair["positive"], max_pixels, None)
            yes_logprob, _no_logprob = _answer_logprobs(policy_model, model_inputs, functional, yes_token_id, no_token_id)
            example_loss = -yes_logprob / len(train_pairs)
            example_loss.backward()
            per_example_sft_losses.append(float(example_loss.detach().cpu()))
        optimizer.step()
        sft_losses.append(sum(per_example_sft_losses))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log_stage("sft_total", time.monotonic() - sft_start, units=len(sft_losses) * len(train_pairs))

    sft_state = {name: value.detach().clone() for name, value in policy_model.named_parameters() if value.requires_grad}
    torch.save(sft_state, output_root / "sft_lora_state.pt")

    reference_probe_inputs = _build_inputs(processor, policy_model, catalog, train_pairs[0]["history"], train_pairs[0]["positive"], max_pixels, None)
    reference_logprob_first = float(_reference_answer_logprobs(policy_model, reference_probe_inputs, torch, functional, yes_token_id, no_token_id)[0].cpu())
    reference_logprob_second = float(_reference_answer_logprobs(policy_model, reference_probe_inputs, torch, functional, yes_token_id, no_token_id)[0].cpu())
    if not math.isclose(reference_logprob_first, reference_logprob_second, abs_tol=1e-5):
        raise RuntimeError("reference model is not reproducible across calls; NoDO must never perturb it")

    portable_catalog = {
        item_id: {**record, "image_path": str(Path(record["image_path"]).relative_to(output_root))}
        for item_id, record in catalog.items()
    }
    bundle = {
        "train": train_pairs,
        "evaluation": evaluation_rows,
        "catalog": portable_catalog,
        "item_order": item_order,
        "lambda_sem": lambda_sem_values,
        "lambda_cf": lambda_cf_values,
        "sft_losses": sft_losses,
        "stage_timings": stage_timings,
        "reference_reproducibility_check": {"first": reference_logprob_first, "second": reference_logprob_second},
    }
    (output_root / "sft_bundle.json").write_text(json.dumps(bundle, indent=2))
    return bundle


def run_arm(
    config: dict[str, Any],
    sft_dir: Path,
    weight: float,
    image_condition: str,
    output_root: Path,
    hanorec_root: Path,
) -> dict[str, Any]:
    """Real DPO training + reranking for exactly one (weight, image_condition)
    arm, resuming from a prior run_sft() push's output mounted at sft_dir
    (a real Kaggle kernel_sources input directory, or a local path when
    testing). Writes arm_w{weight}_{image_condition}.pt and arm_result.json."""
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    stage_timings: list[dict[str, Any]] = []

    def log_stage(name: str, seconds: float, units: int | None = None) -> None:
        payload: dict[str, Any] = {"stage": name, "seconds": seconds}
        if units:
            payload["units"] = units
            payload["seconds_per_unit"] = seconds / units
        stage_timings.append(payload)
        print(json.dumps({"timing": payload}))

    start_time = time.monotonic()
    deadline = start_time + float(config.get("budget_seconds", 6600))

    def remaining() -> float:
        return deadline - time.monotonic()

    bundle = json.loads((sft_dir / "sft_bundle.json").read_text())
    train_pairs = bundle["train"]
    evaluation_rows = bundle["evaluation"]
    catalog = {
        item_id: {**record, "image_path": str(sft_dir / record["image_path"])}
        for item_id, record in bundle["catalog"].items()
    }
    item_order = bundle["item_order"]
    lambda_sem_values = bundle["lambda_sem"]
    lambda_cf_values = bundle["lambda_cf"]

    model_load_start = time.monotonic()
    processor, policy_model = _load_base_model(config)
    policy_model = _wrap_lora(policy_model)
    log_stage("model_load", time.monotonic() - model_load_start)

    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    sft_state = torch.load(sft_dir / "sft_lora_state.pt", map_location=next(policy_model.parameters()).device)

    arm_result = _run_dpo_arm(
        config, processor, policy_model, catalog, train_pairs, evaluation_rows,
        lambda_sem_values, lambda_cf_values, item_order, weight, image_condition,
        yes_token_id, no_token_id, sft_state, remaining, log_stage,
    )

    arm_checkpoint_path = output_root / f"arm_w{weight}_{image_condition}.pt"
    torch.save(arm_result.pop("arm_state"), arm_checkpoint_path)
    arm_result["checkpoint"] = str(arm_checkpoint_path)
    arm_result["stage_timings"] = stage_timings
    arm_result["elapsed_seconds"] = time.monotonic() - start_time

    (output_root / "arm_result.json").write_text(json.dumps(arm_result, indent=2))
    return arm_result


# --------------------------------------------------------------------------
# Single-push entry point (kept for runs small enough to fit one real
# Kaggle session -- the technical smoke test and the calibration run both
# used this path; see phase-02-full-dataset-scaling.md for when to switch
# to run_sft()/run_arm() instead).
# --------------------------------------------------------------------------

def train_and_evaluate(config: dict[str, Any], prepared: dict[str, Any], output_root: Path, hanorec_root: Path) -> dict[str, Any]:
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    stage_timings: list[dict[str, Any]] = []

    def log_stage(name: str, seconds: float, units: int | None = None) -> None:
        payload: dict[str, Any] = {"stage": name, "seconds": seconds}
        if units:
            payload["units"] = units
            payload["seconds_per_unit"] = seconds / units
        stage_timings.append(payload)
        print(json.dumps({"timing": payload}))

    start_time = time.monotonic()
    deadline = start_time + float(config.get("budget_seconds", 6600))

    def remaining() -> float:
        return deadline - time.monotonic()

    catalog = prepared["catalog"]
    train_pairs = prepared["train"]
    evaluation_rows = prepared["evaluation"]

    sft_bundle = run_sft(config, prepared, output_root, hanorec_root)
    lambda_sem_values = sft_bundle["lambda_sem"]
    lambda_cf_values = sft_bundle["lambda_cf"]
    item_order = sft_bundle["item_order"]
    sft_losses = sft_bundle["sft_losses"]
    stage_timings.extend(sft_bundle["stage_timings"])
    reference_logprob_first = sft_bundle["reference_reproducibility_check"]["first"]
    reference_logprob_second = sft_bundle["reference_reproducibility_check"]["second"]

    model_load_start = time.monotonic()
    processor, policy_model = _load_base_model(config)
    policy_model = _wrap_lora(policy_model)
    log_stage("model_load", time.monotonic() - model_load_start)
    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    sft_state = torch.load(output_root / "sft_lora_state.pt", map_location=next(policy_model.parameters()).device)

    arms_run: list[dict[str, Any]] = []
    for weight in config["weights"]:
        for image_condition in ("real", "shuffle"):
            if remaining() <= 0:
                arms_run.append({"weight": weight, "image_condition": image_condition, "status": "SKIPPED_BUDGET"})
                continue
            arm_result = _run_dpo_arm(
                config, processor, policy_model, catalog, train_pairs, evaluation_rows,
                lambda_sem_values, lambda_cf_values, item_order, weight, image_condition,
                yes_token_id, no_token_id, sft_state, remaining, log_stage,
            )
            arm_checkpoint_path = output_root / f"arm_w{weight}_{image_condition}.pt"
            torch.save(arm_result.pop("arm_state"), arm_checkpoint_path)
            arm_result["checkpoint"] = str(arm_checkpoint_path)
            arms_run.append(arm_result)

    (output_root / "hanorec_cf_hardness_result.json").write_text(
        json.dumps(
            {
                "sft_losses": sft_losses,
                "stage_timings": stage_timings,
                "lambda_sem": lambda_sem_values,
                "lambda_cf": lambda_cf_values,
                "arms": arms_run,
                "reference_reproducibility_check": {
                    "first": reference_logprob_first,
                    "second": reference_logprob_second,
                },
                "elapsed_seconds": time.monotonic() - start_time,
                "declared_deviations": [
                    "technical subset: HaRS Top-K computed only over the train-referenced catalog subset, "
                    "not the full Games catalog",
                    "tiny step counts (sft_steps/dpo_steps from config) for a 2 GPU-hour smoke budget, "
                    "not the paper's 5-epoch recipe",
                    "4-bit quantized Qwen2.5-VL-3B with float16 compute dtype, not the paper's full-precision setup",
                    "shuffle image condition is a global item-id permutation across the tiny train-referenced "
                    "subset, disclosed as technical-only, not frequency-matched",
                    "reranking uses per-candidate yes/no logit-margin scoring, not a pairwise tournament",
                    "Eq. 8 responsiveness/beta is computed from a no_grad statistics pass without NoDO noise "
                    "(kept memory-bounded on a single T4); the training pass applies NoDO independently per example",
                ],
            },
            indent=2,
        )
    )

    return {
        "sft_losses": sft_losses,
        "lambda_sem": lambda_sem_values,
        "lambda_cf": lambda_cf_values,
        "arms": arms_run,
        "reference_reproducibility_check": {"first": reference_logprob_first, "second": reference_logprob_second},
    }




# --------------------------------------------------------------------------
# Orchestration: single-arm push (script kernel, resumes from a prior
# run_sft() push mounted via kernel_sources). Real per-arm cost from
# calibration coefficients ~= 1.78h; soft budget below leaves real margin
# under Kaggle's confirmed ~21,600s (6h) hard execution cap.
# --------------------------------------------------------------------------
from pathlib import Path as _Path
import time as _time

_KERNEL_START = _time.monotonic()

EXPERIMENT_CONFIG = {'seed': 2024, 'top_m': 20, 'train_pairs': 530, 'eval_users': 265, 'history_items': 3, 'sft_steps': 2, 'dpo_steps': 2, 'batch_size': 4, 'learning_rate': 0.0001, 'model_id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'model_revision': '66285546d2b821cf421d4f5eb2576359d3770cd3', 'weights': [1.0, 0.0, 0.5], 'noise_sigma': 0.05, 'beta0': 0.1, 'max_pixels': 50176, 'checkpoint_name': 'LLM2Rec-budgeted-|kaggle|working|LLM2Rec|repeated_evaluate_with_seq-Sep-09-2026_11-12-13-c93f12.pth', 'embedding_name': 'Qwen2-0.5B-LLM2Rec-IEM-budgeted_step500_title_item_embs.npy', 'purpose': 'Option B real run (530 train pairs / 265 eval users), resumable per-arm split after 2 confirmed CANCEL_ACKNOWLEDGED session cutoffs on monolithic notebook pushes', 'artifact_pins': {'train_data.txt': '1300e5deec29d4bede6e32bfa1a5eade563f53a6916da95d67754ec40b9475b8', 'val_data.txt': '484a97cfbcff2a78e92612b15863ce08c84c285c7bfb325ec1f33d3884e6de16', 'test_data.txt': '80074ffea37928d92c35038e9d8bbf3de0cfcf01308ae4173e3c680c1bc50d1f', 'item_titles.json': '17f501809c4159905cfa6dfa5626d3d52d0be32bc9747ee89b003662f6f5e84c', 'LLM2Rec-budgeted-|kaggle|working|LLM2Rec|repeated_evaluate_with_seq-Sep-09-2026_11-12-13-c93f12.pth': 'fb0d1d3916f10b5b6f0d9f597e7b2ac8437f445c15629ebf3099333b72806a59', 'Qwen2-0.5B-LLM2Rec-IEM-budgeted_step500_title_item_embs.npy': '89f9d3dd17f9ec49537f910e39c9761878299f7d24abdcf805ab0db91433c0f0', 'games_sasrec_step500.log': 'a60181a26114072dd5442fd9ffe4b338505010d8b321d9e31ca1646b341ff154', 'Video_Games_image_manifest.jsonl': '5d9c586d2d05c6ff208c7a0f945a441a4fc60ef0d18ec8aacc269fbf67a42d04'}}
EXPERIMENT_CONFIG["budget_seconds"] = 9000.0

WEIGHT = 0.5
IMAGE_CONDITION = "real"

# Locate the mounted run_sft() output by filename glob, not an assumed mount
# directory name (kernel_sources mount paths vary), matching the proven
# pattern in experiments/multimodal_llm_rs/kaggle/llm2rec-caption-augmentation/full-corpus-v2-inline/runner.ipynb.
_sft_candidates = sorted(_Path("/kaggle/input").glob("**/sft_bundle.json"))
if not _sft_candidates:
    raise RuntimeError("no mounted run_sft() output found under /kaggle/input -- check kernel_sources")
SFT_DIR = _sft_candidates[0].parent
print({"stage": "sft_dir_located", "sft_dir": str(SFT_DIR), "elapsed": _time.monotonic() - _KERNEL_START})

OUTPUT_ROOT = _Path("/kaggle/working/hanorec_cf_hardness_output")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print({"stage": "run_arm_start", "weight": WEIGHT, "image_condition": IMAGE_CONDITION, "elapsed": _time.monotonic() - _KERNEL_START})
arm_result = run_arm(EXPERIMENT_CONFIG, SFT_DIR, WEIGHT, IMAGE_CONDITION, OUTPUT_ROOT, _Path("/kaggle/working"))
print({
    "stage": "run_arm_complete",
    "elapsed": _time.monotonic() - _KERNEL_START,
    "status": arm_result["status"],
    "mean_ndcg@10": arm_result["mean_ndcg@10"],
    "mean_recall@10": arm_result["mean_recall@10"],
})
