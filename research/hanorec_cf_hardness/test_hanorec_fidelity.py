from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train as train_module  # noqa: E402
from train import (  # noqa: E402
    _accumulation_group_size,
    _cuda_peak_memory,
    _build_dpo_examples,
    _chunked,
    _finite,
    _json_default,
    _reset_cuda_peak_memory,
    _score_rows,
    _restore_active_adapters,
    _set_active_adapters,
    _set_only_trainable_adapter,
    hars_normalize_hardness,
    responsiveness,
)

class _AdapterBase:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def set_adapter(self, adapters):
        names = [adapters] if isinstance(adapters, str) else list(adapters)
        self.calls.append(names)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = any(f".{adapter}." in name for adapter in names)

class _Parameter:
    def __init__(self):
        self.requires_grad = False


class _AdapterModel:
    def __init__(self):
        self.calls = []
        self.active_adapter = None
        self.peft_config = {"sft": object(), "dpo": object()}
        self.parameters = {
            "layer.sft.lora_A": _Parameter(),
            "layer.dpo.lora_A": _Parameter(),
        }
        self.base_model = _AdapterBase(self)

    def set_adapter(self, adapter):
        self.calls.append(adapter)

    def named_parameters(self):
        return self.parameters.items()

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]


class _CudaProperties:
    total_memory = 1_000


class _FakeCuda:
    def __init__(self):
        self.reset = False

    def is_available(self):
        return True

    def synchronize(self):
        return None

    def reset_peak_memory_stats(self):
        self.reset = True

    def current_device(self):
        return 0

    def get_device_properties(self, _device):
        return _CudaProperties()

    def get_device_name(self, _device):
        return "fake-gpu"

    def max_memory_allocated(self, _device):
        return 600

    def max_memory_reserved(self, _device):
        return 750


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()



class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _ScoreTorch:
    def no_grad(self):
        return _NoGrad()


class _ScoreModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _ScoreValue:
    def __init__(self, value):
        self.value = float(value)

    def __sub__(self, other):
        return _ScoreValue(self.value - other.value)

    def cpu(self):
        return self.value
class _Scalar:
    def item(self):
        return 1.5


class HaNoRecFidelityTests(unittest.TestCase):
    def test_upstream_responsiveness_has_global_trimmed_batch_semantics(self) -> None:
        value = responsiveness([1.0, 2.0, 3.0, 100.0])
        self.assertGreater(value, 0.0)
        self.assertLess(value, 2.0)
        with self.assertRaises(ValueError):
            responsiveness([1.0, float("nan"), 3.0])

    def test_dpo_examples_preserve_binary_pair_order(self) -> None:
        examples = _build_dpo_examples(
            [{"history": [1, 2, 3], "positive": 4, "negative": 5}],
            yes_token_id=10,
            no_token_id=11,
            shuffle_map=None,
        )
        self.assertEqual(
            [(row["candidate"], row["chosen_token"], row["rejected_token"]) for row in examples],
            [(4, 10, 11), (5, 11, 10)],
        )

    def test_batch_boundaries_and_hardness_are_fail_closed(self) -> None:
        self.assertEqual([len(batch) for batch in _chunked(list(range(10)), 4, True)], [4, 4])
        self.assertEqual([len(batch) for batch in _chunked(list(range(10)), 4, False)], [4, 4, 2])
        self.assertEqual(len(hars_normalize_hardness([0.0, 1.0, 2.0])), 3)
        with self.assertRaises(ValueError):
            _chunked([1, 2], 0, True)
        with self.assertRaises(ValueError):
            _finite([], "values")

    def test_multi_adapter_activation_freezes_sft_parameters(self) -> None:
        model = _AdapterModel()
        _set_active_adapters(model, ["sft", "dpo"])
        _set_only_trainable_adapter(model, "dpo")
        self.assertFalse(model.parameters["layer.sft.lora_A"].requires_grad)
        self.assertTrue(model.parameters["layer.dpo.lora_A"].requires_grad)
        _restore_active_adapters(model, ["sft", "dpo"])
        self.assertFalse(model.parameters["layer.sft.lora_A"].requires_grad)
        self.assertTrue(model.parameters["layer.dpo.lora_A"].requires_grad)

    def test_multi_adapter_activation_uses_peft_base_model(self) -> None:
        model = _AdapterModel()
        _set_active_adapters(model, ["sft", "dpo"])
        self.assertEqual(model.base_model.calls, [["sft", "dpo"]])
        self.assertEqual(model.active_adapter, "dpo")
        self.assertIs(model.active_peft_config, model.peft_config["dpo"])
        _set_active_adapters(model, "sft")
        self.assertEqual(model.calls, ["sft"])
    def test_partial_accumulation_uses_one_group_divisor(self) -> None:
        self.assertEqual(
            [_accumulation_group_size(position, 10, 4) for position in range(10)],
            [4, 4, 4, 4, 4, 4, 4, 4, 2, 2],
        )
        with self.assertRaises(ValueError):
            _accumulation_group_size(0, 0, 4)


    def test_cuda_peak_memory_reports_capacity_fractions(self) -> None:
        torch_module = _FakeTorch()
        _reset_cuda_peak_memory(torch_module)
        self.assertTrue(torch_module.cuda.reset)
        self.assertEqual(
            _cuda_peak_memory(torch_module),
            {
                "device": "fake-gpu",
                "total_bytes": 1_000,
                "peak_allocated_bytes": 600,
                "peak_reserved_bytes": 750,
                "peak_allocated_fraction": 0.6,
                "peak_reserved_fraction": 0.75,
            },
        )

    def test_candidate_scoring_batches_without_changing_rank_contract(self) -> None:
        model = _ScoreModel()
        batch_sizes = []

        def fake_inputs(_processor, _model, _catalog, examples, _pixels, _shuffle):
            batch_sizes.append(len(examples))
            return examples

        def fake_logprobs(_model, examples, _functional, _yes, _no):
            return [(_ScoreValue(example["candidate"]), _ScoreValue(0.0)) for example in examples]

        with (
            patch.object(train_module, "_build_batch_inputs", side_effect=fake_inputs),
            patch.object(train_module, "_batch_answer_logprobs", side_effect=fake_logprobs),
        ):
            result = _score_rows(
                [{
                    "user_id": "u",
                    "history": [9, 8, 7],
                    "target": 2,
                    "candidates": [1, 3, 2],
                }],
                object(), model, {}, 8192, 1, 0, _ScoreTorch(), object(), None,
                eval_batch_size=2,
            )

        self.assertEqual(batch_sizes, [2, 1])
        self.assertEqual(result[0]["ranked_candidates"], [3, 2, 1])
        self.assertEqual(result[0]["rank"], 2)
        self.assertTrue(model.training)

    def test_json_converter_handles_scalar_protocol_and_rejects_unknown_values(self) -> None:
        self.assertAlmostEqual(_json_default(_Scalar()), 1.5)
        with self.assertRaises(TypeError):
            _json_default(object())


if __name__ == "__main__":
    unittest.main()
