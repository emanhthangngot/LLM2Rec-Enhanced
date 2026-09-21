"""Captioner, paraphraser, and tokenizer backends.

Real backends (Florence-2, Qwen2.5 paraphraser, Qwen tokenizer) import
torch/transformers/PIL/huggingface_hub lazily inside their methods, never at
module import time, so this module imports cleanly on a CPU-only box with no
GPU stack installed (this workstation has none: see
plans/260915-0955-visual-delta-fusion-pilot/plan.md "Evidence and unresolved
risks"). Stub backends are what local/CPU tests exercise; they are never
substituted for the real backends on Kaggle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

RUNTIME_DEPENDENCIES = {"transformers": "4.44.2"}


def ensure_runtime_dependencies(expected: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pin `transformers` to the version proven compatible with Florence-2's
    custom config code across this research family's completed Kaggle runs
    (`ensure_runtime_dependencies` in
    experiments/multimodal_llm_rs/kaggle/llm2rec-visual/visual_signal.py).

    Without this pin, the Kaggle image's newer default `transformers`
    raises `AttributeError: 'Florence2LanguageConfig' object has no
    attribute 'forced_bos_token_id'` inside Florence-2's own
    `configuration_florence2.py`, for both the primary and fallback caption
    model revisions alike (see
    plans/260915-0955-visual-delta-fusion-pilot/ISSUES.md #10). GPU/Kaggle
    only; call this before constructing `Florence2Captioner` or
    `QwenParaphraser`.
    """
    import importlib.metadata
    import subprocess
    import sys

    resolved_expected = dict(expected) if expected is not None else dict(RUNTIME_DEPENDENCIES)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", *(f"{name}=={version}" for name, version in resolved_expected.items())],
        check=True,
    )
    resolved = {name: importlib.metadata.version(name) for name in resolved_expected}
    if resolved != resolved_expected:
        raise RuntimeError(f"dependency resolution mismatch: expected={resolved_expected} resolved={resolved}")
    return resolved


CAPTION_MODEL_ID = "microsoft/Florence-2-large"
CAPTION_TASK_PROMPT = "<CAPTION>"
CAPTION_MAX_NEW_TOKENS = 128
CAPTION_NUM_BEAMS = 3
# Last commit before the 2024-12-08 continued-pretrain weight swap: original
# FLD-5B weights matching the model card's published benchmark table, no
# "might not be trained well" caveat. See
# plans/260915-0955-visual-delta-fusion-pilot/ISSUES.md #7 for the full
# commit-history evidence behind this pin.
CAPTION_MODEL_REVISION = "f0acedbf9b780e04fe1f9111fcf53187388f3d03"
# Current `main`: continued-pretrained 4k-context checkpoint, card-flagged as
# possibly undertrained. Only used if the primary revision fails to load.
CAPTION_MODEL_REVISION_FALLBACK = "21a599d414c4d928c9032694c424fb94458e3594"

PARAPHRASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PARAPHRASE_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PARAPHRASE_MAX_NEW_TOKENS = 128
PARAPHRASE_PROMPT_TEMPLATE = (
    "Restate only the supplied product title. Preserve named entities and "
    "attributes. Add no facts or recommendations. Return one short sentence. "
    "Title: {title}"
)

TOKENIZER_MODEL_ID = "Qwen/Qwen2-0.5B"
TOKENIZER_MODEL_REVISION = "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"


@dataclass(frozen=True)
class CaptionResult:
    """One raw generation result, kept separate from any later cleaning."""

    item_id: int
    raw_text: str
    status: str  # "ok" | "empty" | "decode_failure"
    generated_token_count: int | None = None


class Captioner(Protocol):
    def caption_batch(self, items: Sequence[tuple[int, Path]]) -> list[CaptionResult]: ...


class Paraphraser(Protocol):
    def paraphrase_batch(self, items: Sequence[tuple[int, str]]) -> list[CaptionResult]: ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class StubCaptioner:
    """Returns canned captions from a fixed dict. Used only by local tests."""

    def __init__(self, captions_by_item: Mapping[int, str]) -> None:
        self._captions = dict(captions_by_item)

    def caption_batch(self, items: Sequence[tuple[int, Path]]) -> list[CaptionResult]:
        results = []
        for item_id, _path in items:
            text = self._captions.get(item_id, "")
            status = "ok" if text else "empty"
            results.append(CaptionResult(item_id=item_id, raw_text=text, status=status))
        return results


class StubParaphraser:
    """Returns canned paraphrases from a fixed dict. Used only by local tests."""

    def __init__(self, paraphrases_by_item: Mapping[int, str]) -> None:
        self._paraphrases = dict(paraphrases_by_item)

    def paraphrase_batch(self, items: Sequence[tuple[int, str]]) -> list[CaptionResult]:
        results = []
        for item_id, _title in items:
            text = self._paraphrases.get(item_id, "")
            status = "ok" if text else "empty"
            results.append(CaptionResult(item_id=item_id, raw_text=text, status=status))
        return results


class StubTokenizer:
    """One token per whitespace-separated word. Used only by local tests."""

    def encode(self, text: str) -> list[int]:
        return text.split()


CAPTION_ATTN_IMPLEMENTATION = "sdpa"


def _skip_flash_attn_static_scan():
    """Context manager neutralizing transformers' static `flash_attn` import
    scan for Florence-2's remote modeling file, without touching real imports.

    Evidence (plans/260915-0955-visual-delta-fusion-pilot/ISSUES.md #12):
    Florence-2's ``modeling_florence2.py`` imports ``flash_attn`` only inside
    ``if is_flash_attn_2_available(): ...`` guards and dispatches attention
    via ``FLORENCE2_ATTENTION_CLASSES[config._attn_implementation]``; an
    ``sdpa``/``eager`` configuration never executes that branch. But
    ``transformers.dynamic_module_utils.check_imports`` performs a *static*
    source scan of every top-level import statement and raises
    ``ImportError`` for ``flash_attn`` regardless of the guard, even though
    Kaggle's T4 (sm_75) and P100 (sm_60) accelerators are both below the
    sm_80 FlashAttention-2 requires — so real installation is neither
    possible to use nor the fix. This patches only
    ``transformers.dynamic_module_utils.get_imports`` (the function the
    static scan reads its package list from) to drop ``"flash_attn"``,
    scoped to this context only, and requires the caller to pass an explicit
    non-flash ``attn_implementation`` so the omission is never silent.
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    import transformers.dynamic_module_utils as dynamic_module_utils

    original_get_imports = dynamic_module_utils.get_imports

    def get_imports_without_flash_attn(filename):
        imports = original_get_imports(filename)
        return [name for name in imports if name != "flash_attn"]

    @contextmanager
    def _cm():
        with patch.object(dynamic_module_utils, "get_imports", get_imports_without_flash_attn):
            yield

    return _cm()


class Florence2Captioner:
    """Real Florence-2-large captioner. GPU/Kaggle only.

    Every heavy import (torch, transformers, PIL, huggingface_hub) is inside
    ``__init__``/``caption_batch`` so importing this module never requires
    them. ``resolved_revision`` is pinned once at construction via
    ``huggingface_hub.model_info`` and reused for every batch, never
    re-resolved from ``main`` per call (phase-01 F-requirement: "Select
    immutable HF revision once"). Loads with an explicit
    ``attn_implementation="sdpa"`` and a scoped static-scan patch (see
    ``_skip_flash_attn_static_scan``) so a genuinely unused ``flash_attn``
    import in the remote modeling file never blocks loading on
    non-Ampere Kaggle accelerators.
    """

    def __init__(self, model_id: str = CAPTION_MODEL_ID, revision: str | None = CAPTION_MODEL_REVISION) -> None:
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_id = model_id
        self.resolved_revision = revision or model_info(model_id).sha
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        with _skip_flash_attn_static_scan():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=self.resolved_revision,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                attn_implementation=CAPTION_ATTN_IMPLEMENTATION,
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(
                model_id, revision=self.resolved_revision, trust_remote_code=True
            )

    # Internal GPU-memory safety cap. num_beams=3 beam search internally
    # expands the effective batch by 3x during generation; a naive full
    # 64-image batch (the smoke-test call shape) measured 13.35 GiB in use
    # against a T4's 14.56 GiB before even allocating the attention matrix,
    # producing a real `CUDA out of memory` error (issue #25 verification
    # run). Every caller — this fixed constant, not the caller's list size —
    # controls the actual forward-pass batch, so existing 64/300-item
    # callers (`real_generator_smoke`, `quality_review`) stay correct.
    MAX_GPU_BATCH = 4

    def caption_batch(self, items: Sequence[tuple[int, Path]]) -> list[CaptionResult]:
        import torch
        from PIL import Image

        results: list[CaptionResult] = []
        valid_ids: list[int] = []
        valid_images: list["Image.Image"] = []
        for item_id, image_path in items:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                results.append(CaptionResult(item_id=item_id, raw_text="", status="decode_failure"))
                continue
            valid_ids.append(item_id)
            valid_images.append(image)
        for start in range(0, len(valid_images), self.MAX_GPU_BATCH):
            chunk_ids = valid_ids[start:start + self.MAX_GPU_BATCH]
            chunk_images = valid_images[start:start + self.MAX_GPU_BATCH]
            # Every image shares the identical `<CAPTION>` prompt, so the
            # text side never needs padding; only pixel_values stack.
            inputs = self.processor(
                text=[CAPTION_TASK_PROMPT] * len(chunk_images), images=chunk_images, return_tensors="pt", padding=True
            ).to(self.device, self.torch_dtype)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=CAPTION_MAX_NEW_TOKENS,
                    num_beams=CAPTION_NUM_BEAMS,
                    do_sample=False,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            pad_id = self.model.generation_config.pad_token_id
            for item_id, image, row in zip(chunk_ids, chunk_images, generated_ids):
                # `generate()` right-pads every row in the batch to the
                # longest generated sequence with `pad_token_id`. Trim only
                # that trailing padding (never a legitimate generated
                # token) so each row is decoded with the exact same
                # `skip_special_tokens=False` semantics the proven
                # single-item path used (issue #13's verified
                # `status: PASS` run) — batching must not change what
                # `post_process_generation` receives for any one image.
                end = row.shape[-1]
                if pad_id is not None:
                    while end > prompt_length and int(row[end - 1]) == pad_id:
                        end -= 1
                trimmed_row = row[:end]
                generated_text = self.processor.batch_decode(trimmed_row.unsqueeze(0), skip_special_tokens=False)[0]
                parsed = self.processor.post_process_generation(
                    generated_text, task=CAPTION_TASK_PROMPT, image_size=(image.width, image.height)
                )
                raw_text = str(parsed.get(CAPTION_TASK_PROMPT, "")).strip()
                status = "ok" if raw_text else "empty"
                results.append(CaptionResult(
                    item_id=item_id, raw_text=raw_text, status=status,
                    generated_token_count=int(end - prompt_length),
                ))
            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results


class QwenParaphraser:
    """Real title-only paraphrase control. GPU/Kaggle only.

    Sees only the title text — never the image, caption, or any interaction
    data — so it cannot leak visual information into the paraphrase control
    (plan.md: "a separate frozen text model used ONLY to build a control").
    """

    def __init__(self, model_id: str = PARAPHRASE_MODEL_ID, revision: str | None = PARAPHRASE_MODEL_REVISION) -> None:
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.resolved_revision = revision or model_info(model_id).sha
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.resolved_revision)
        # Left padding is required for correct batched causal-LM generation:
        # it keeps every sequence's real content right-aligned, so the
        # generated continuation starts at the same column index for every
        # row regardless of prompt length.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=self.resolved_revision, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device).eval()

    # Same GPU-memory safety rationale as `Florence2Captioner.MAX_GPU_BATCH`:
    # this fixed constant, not the caller's list size, controls the actual
    # forward-pass batch, so existing 64/300-item callers stay correct.
    MAX_GPU_BATCH = 4

    def paraphrase_batch(self, items: Sequence[tuple[int, str]]) -> list[CaptionResult]:
        import torch

        results: list[CaptionResult] = []
        for start in range(0, len(items), self.MAX_GPU_BATCH):
            chunk = items[start:start + self.MAX_GPU_BATCH]
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": PARAPHRASE_PROMPT_TEMPLATE.format(title=title)}],
                    add_generation_prompt=True, tokenize=False,
                )
                for _item_id, title in chunk
            ]
            encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **encoded, max_new_tokens=PARAPHRASE_MAX_NEW_TOKENS, do_sample=False
                )
            # Left padding means every prompt (real content plus left pad)
            # occupies exactly `encoded["input_ids"].shape[-1]` columns, so
            # the newly generated continuation is that same uniform suffix
            # for every row in the batch — the standard batched-generation
            # slicing pattern, unaffected by each prompt's real (unpadded)
            # length.
            prompt_length = encoded["input_ids"].shape[-1]
            new_tokens_batch = generated_ids[:, prompt_length:]
            pad_id = self.tokenizer.pad_token_id
            for (item_id, _title), new_tokens in zip(chunk, new_tokens_batch):
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                status = "ok" if text else "empty"
                token_count = int((new_tokens != pad_id).sum()) if pad_id is not None else int(new_tokens.shape[-1])
                results.append(CaptionResult(
                    item_id=item_id, raw_text=text, status=status,
                    generated_token_count=token_count,
                ))
            del encoded, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results


class QwenTokenizer:
    """Real Qwen2 tokenizer wrapper for token-cap budgeting. GPU/Kaggle only."""

    def __init__(self, model_id: str = TOKENIZER_MODEL_ID, revision: str | None = TOKENIZER_MODEL_REVISION) -> None:
        from huggingface_hub import model_info
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.resolved_revision = revision or model_info(model_id).sha
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.resolved_revision)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)
