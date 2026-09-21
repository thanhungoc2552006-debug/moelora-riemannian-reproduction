"""Check recipe execution, provenance and protection of existing runs."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from moelora_repro.run import ENTRYPOINTS, load_recipe, main, parameter_args, run_recipe


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_checked_in_recipes_and_driver_flags(self):
        recipes = sorted((ROOT / "configs").glob("*/*.json"))
        recipes += sorted((ROOT / "experiments/synthetic/configs").glob("*.json"))
        self.assertTrue(recipes, "No checked-in recipes were discovered")
        for path in recipes:
            with self.subTest(recipe=path.name):
                recipe = load_recipe(path)
                # Help exits before loading a dataset/model, but catches import failures.
                result = subprocess.run(
                    [sys.executable, "-m", ENTRYPOINTS[recipe["entrypoint"]],
                     *parameter_args(recipe["parameters"]), "--help"],
                    text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--out", result.stdout)
                for flag in parameter_args(recipe["parameters"]):
                    if flag.startswith("--"):
                        self.assertIn(flag, result.stdout)

    def test_overrides_and_dry_run_do_not_create_outputs(self):
        config = ROOT / "experiments/synthetic/configs/main.json"
        recipe = load_recipe(config, ["steps=3", "device=cpu", "seeds=[5,6]"])
        self.assertEqual(recipe["parameters"]["steps"], 3)
        self.assertEqual(recipe["parameters"]["seeds"], [5, 6])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "untouched"
            code = main(["--config", str(config), "--out", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            load_recipe(config, ["out=somewhere"])
        with self.assertRaises(ValueError):
            load_recipe(config, ["steps"])

    def test_successful_run_records_actual_config_and_refuses_overwrite(self):
        recipe = {
            "name": "test-run", "entrypoint": "synthetic-run",
            "parameters": {"steps": 2, "train_size": 16, "val_size": 8,
                           "hidden_dim": 4, "output_dim": 2, "num_experts": 2,
                           "rank": 1, "batch_size": 4, "diagnostic_size": 8,
                           "top_k": 1, "seed": 7, "device": "cpu"},
        }
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(temp) / "config.json"
            source.write_text(json.dumps(recipe))
            output = Path(temp) / "new-run"
            self.assertEqual(run_recipe(recipe, output, source), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["exit_code"], 0)
            self.assertIsNotNone(manifest["finished_at"])
            self.assertIn("torch", manifest["packages"])
            self.assertEqual(json.loads((output / "recipe.json").read_text()), recipe)
            metadata = json.loads((output / "riemannian_k1_seed7.json").read_text())
            self.assertEqual(metadata["seed"], 7)
            self.assertEqual(metadata["config"]["steps"], 2)
            before = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_recipe(recipe, output, source)
            self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_failed_driver_preserves_failure_and_log(self):
        recipe = {"name": "invalid-run", "entrypoint": "synthetic-run",
                  "parameters": {"steps": 0, "device": "cpu"}}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(temp) / "config.json"
            source.write_text(json.dumps(recipe))
            output = Path(temp) / "failed-run"
            code = run_recipe(recipe, output, source)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotEqual(code, 0)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["exit_code"], code)
            self.assertIn("ValueError", (output / "console.log").read_text())


if __name__ == "__main__":
    unittest.main()
