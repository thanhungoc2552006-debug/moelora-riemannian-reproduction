"""Check corrected labels, public model API and archived source provenance."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from moelora_repro.scienceqa import train
from moelora_repro.scienceqa.model import MoELoRALinear, inject_moe_lora
from moelora_repro.scienceqa.optimizer import RiemannianSGD
from moelora_repro.run import parameter_args

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    bos_token_id, eos_token_id = 1, 2

    def __call__(self, text, add_special_tokens=True, truncation=False):
        ids = list(range(3, 3 + len(text.split())))
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}


class ScienceQATests(unittest.TestCase):
    def test_long_prompt_preserves_complete_answer_and_eos(self):
        tokenizer = FakeTokenizer()
        example = {"hint": "long " * 500, "question": "Which option?",
                   "choices": ["one", "two"], "answer": 1}
        row = train.encode_training(example, tokenizer)
        prompt = train.prompt_ids(example, tokenizer)
        self.assertEqual(len(prompt), train.PROMPT_BUDGET)
        self.assertEqual(prompt[0], tokenizer.bos_token_id)
        self.assertEqual(row["input_ids"][:len(prompt)], prompt)
        self.assertEqual(row["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(row["labels"][len(prompt):], row["input_ids"][len(prompt):])
        self.assertEqual(row["labels"][-1], tokenizer.eos_token_id)
        self.assertLessEqual(len(row["input_ids"]), train.MAX_LENGTH)
        self.assertTrue(any(x != -100 for x in row["labels"][1:]))

    def test_dropout_is_explicit_and_eval_forward_is_equal(self):
        torch.manual_seed(17)
        a = MoELoRALinear(torch.nn.Linear(5, 3), rank=2, num_experts=3, top_k=2)
        self.assertEqual(a.dropout.p, 0.0)
        with torch.no_grad():
            for b in a.lora_B:
                b.weight.normal_()
        b = copy.deepcopy(a)
        b.mode = "moe-riemannian"
        a.eval()
        b.eval()
        x = torch.randn(2, 4, 5)
        torch.testing.assert_close(a(x), b(x))
        toy = torch.nn.Module()
        toy.q_proj = torch.nn.Linear(5, 3)
        inject_moe_lora(toy, rank=2, num_experts=3, top_k=2, dropout=0.05)
        self.assertEqual(toy.q_proj.dropout.p, 0.05)

    def test_checkpoint_roundtrip_restores_parameters_and_position_on_cpu(self):
        model = torch.nn.Module()
        model.q_proj = torch.nn.Linear(5, 3)
        inject_moe_lora(model, rank=2, num_experts=3, top_k=2)
        gates = torch.optim.SGD(model.q_proj.gate.parameters(), lr=3e-8)
        experts = RiemannianSGD(model, lr=3e-5, reg=1e-6)
        expected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        cfg = {"test": True}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(torch.cuda, "get_rng_state_all", return_value=[]), \
             patch.object(torch.cuda, "set_rng_state_all"):
            out = Path(temp)
            train.save_checkpoint(out, model, gates, experts, cfg, 50, 2, 9)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(1)
            self.assertEqual(train.load_checkpoint(out, model, gates, experts, cfg), (50, 2, 9))
            for name, p in model.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p, expected[name])
            with self.assertRaises(ValueError):
                train.load_checkpoint(out, model, gates, experts, {"changed": True})

    def test_archived_sources_and_predictions_match_summaries(self):
        results = ROOT / "results/scienceqa"
        files = {"driver": "scienceqa_colab_driver.py", "model": "model.py",
                 "optimizer": "optimizer.py"}
        for name in ("rsgd_seed42", "grsgd_seed42"):
            run = results / "runs" / name
            summary = json.loads((run / "summary.json").read_text())
            for key, filename in files.items():
                data = (results / "source_snapshot" / filename).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), summary["config"]["source_hashes"][key])
            predictions = json.loads((run / "test_predictions.json").read_text())
            self.assertEqual(len(predictions), summary["total"])
            self.assertEqual(sum(p["prediction"] == p["gold"] for p in predictions), summary["correct"])
            self.assertEqual(sum(p["prediction"] is None for p in predictions), summary["invalid_predictions"])

    def test_boolean_recipe_flags(self):
        self.assertEqual(parameter_args({"smoke": True, "unused": False, "steps": 2}),
                         ["--smoke", "--steps", "2"])


if __name__ == "__main__":
    unittest.main()
