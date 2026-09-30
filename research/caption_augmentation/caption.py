"""Qwen3-VL captioner, Qwen paraphraser, and tokenizer backends.

Real backends import torch/transformers/PIL lazily so this module stays
importable on CPU-only machines. Qwen caption generation and downstream
LLM2Rec training run in separate Kaggle kernels because their Transformers
requirements differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

RUNTIME_DEPENDENCIES = {"transformers": ">=4.57.0,<5"}
QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3VL_MODEL_REVISION: str | None = None
CAPTION_PROMPT_TEMPLATE = """You see a product image from an online store. The product title is: "{title}".
Describe ONLY visual information that is NOT already stated in the title.
Do not repeat or transcribe any words printed on the product, box, or cover.
Answer in at most 40 words as "key: value" pairs, choosing only relevant keys from:
color, material, shape/form factor, included items, art style, characters or scene shown, mood/theme, target audience cues, condition.
If the image clearly shows a different kind of product than the title, answer exactly: MISMATCH."""
CAPTION_MAX_NEW_TOKENS = 96
CAPTION_BATCH_SIZE = 1

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


def ensure_runtime_dependencies(expected: Mapping[str, str] | None = None) -> dict[str, str]:
    """Install the compatible Qwen caption runtime without changing torch."""
    import importlib.metadata
    import subprocess
    import sys

    resolved_expected = dict(expected) if expected is not None else dict(RUNTIME_DEPENDENCIES)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir",
         *(f"{name}{constraint}" for name, constraint in resolved_expected.items())],
        check=True,
    )
    resolved = {name: importlib.metadata.version(name) for name in resolved_expected}
    from packaging.specifiers import SpecifierSet

    invalid = {
        name: (constraint, resolved[name])
        for name, constraint in resolved_expected.items()
        if resolved[name] not in SpecifierSet(constraint)
    }
    if invalid:
        raise RuntimeError(f"dependency resolution mismatch: {invalid}")
    return resolved


def caption_prompt(title: str) -> str:
    if not title:
        raise ValueError("caption prompt requires a non-empty title")
    return CAPTION_PROMPT_TEMPLATE.format(title=title)


def prompt_sha256() -> str:
    import hashlib

    return hashlib.sha256(CAPTION_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_revision(model_id: str, revision: str | None) -> str:
    if revision:
        return revision
    from huggingface_hub import model_info

    return model_info(model_id).sha


@dataclass(frozen=True)
class CaptionResult:
    """One raw generation result, kept separate from any later cleaning."""

    item_id: int
    raw_text: str
    status: str  # "ok" | "mismatch" | "empty" | "decode_failure"
    generated_token_count: int | None = None


class Captioner(Protocol):
    def caption_batch(self, items: Sequence[tuple[int, str, Path]]) -> list[CaptionResult]: ...


class Paraphraser(Protocol):
    def paraphrase_batch(self, items: Sequence[tuple[int, str]]) -> list[CaptionResult]: ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class StubCaptioner:
    """Returns canned captions from a fixed dict. Used only by local tests."""

    def __init__(self, captions_by_item: Mapping[int, str]) -> None:
        self._captions = dict(captions_by_item)

    def caption_batch(self, items: Sequence[tuple[int, str, Path]]) -> list[CaptionResult]:
        results = []
        for item_id, _title, _path in items:
            text = self._captions.get(item_id, "")
            status = "mismatch" if text.strip().upper() == "MISMATCH" else ("ok" if text else "empty")
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


class Qwen3VLCaptioner:
    """Real title-conditioned Qwen3-VL captioner. GPU/Kaggle only."""

    def __init__(
        self,
        model_id: str = QWEN3VL_MODEL_ID,
        revision: str | None = QWEN3VL_MODEL_REVISION,
        batch_size: int = CAPTION_BATCH_SIZE,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        self.resolved_revision = model_revision(model_id, revision)
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, revision=self.resolved_revision)
        self.processor_revision = self.resolved_revision
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=self.resolved_revision,
            torch_dtype=self.torch_dtype,
            attn_implementation="sdpa",
            device_map={"": 0} if torch.cuda.is_available() else None,
        ).eval()
        self.model_revision = self.resolved_revision

    def caption_batch(self, items: Sequence[tuple[int, str, Path]]) -> list[CaptionResult]:
        import torch
        from PIL import Image

        results: list[CaptionResult] = []
        valid: list[tuple[int, str, "Image.Image"]] = []
        for item_id, title, image_path in items:
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                image.thumbnail((768, 768))
            except Exception:
                results.append(CaptionResult(item_id=item_id, raw_text="", status="decode_failure"))
                continue
            valid.append((item_id, title, image))

        for start in range(0, len(valid), self.batch_size):
            chunk = valid[start:start + self.batch_size]
            conversations = [
                [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": caption_prompt(title)},
                    ],
                }]
                for _item_id, title, image in chunk
            ]
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=CAPTION_MAX_NEW_TOKENS, do_sample=False
                )
            prompt_length = inputs["input_ids"].shape[-1]
            generated = generated_ids[:, prompt_length:]
            for (item_id, _title, _image), token_ids in zip(chunk, generated):
                raw_text = self.processor.batch_decode(
                    token_ids.unsqueeze(0),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                status = (
                    "mismatch" if raw_text.upper() == "MISMATCH"
                    else ("ok" if raw_text else "empty")
                )
                results.append(CaptionResult(
                    item_id=item_id,
                    raw_text=raw_text,
                    status=status,
                    generated_token_count=int(token_ids.shape[-1]),
                ))
            del inputs, generated_ids
            for _item_id, _title, image in chunk:
                image.close()
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

    # Bound the text-control model's GPU batch independently from captioning.
    # This controls forward-pass memory without changing generated text.
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
