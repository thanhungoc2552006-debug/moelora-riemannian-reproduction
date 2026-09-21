"""Run a checked-in experiment recipe and record its provenance."""
import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


ENTRYPOINTS = {
    "synthetic-sweep": "moelora_synthetic.experiment",
    "synthetic-run": "moelora_synthetic.train",
    "scienceqa-pilot": "moelora_repro.scienceqa.train_pilot",
    "scienceqa-long": "moelora_repro.scienceqa.train",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_recipe(path, overrides=()):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON object")
    required = {"name", "entrypoint", "parameters"}
    if set(recipe) != required:
        raise ValueError(f"Recipe must contain exactly {sorted(required)}")
    if not isinstance(recipe["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", recipe["name"]):
        raise ValueError("Recipe name must contain lowercase letters, digits, '-' or '_'")
    if recipe["entrypoint"] not in ENTRYPOINTS:
        raise ValueError(f"Unknown entrypoint; choose from {list(ENTRYPOINTS)}")
    if not isinstance(recipe["parameters"], dict):
        raise ValueError("Recipe parameters must be an object")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Overrides must be KEY=VALUE, for example device=cpu")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        recipe["parameters"][key] = value
    # Validate before reserving an output directory or starting a subprocess.
    parameter_args(recipe["parameters"])
    return recipe


def parameter_args(parameters):
    arguments = []
    for key, value in parameters.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key == "out":
            raise ValueError(f"Invalid parameter {key!r}; use --out for output directories")
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        if not values or any(type(item) not in (str, int, float) for item in values):
            raise ValueError(f"Parameter {key!r} must be a scalar or a nonempty list of scalars")
        arguments.append("--" + key.replace("_", "-"))
        arguments.extend(str(item) for item in values)
    return arguments


def git_metadata(directory):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def package_versions():
    versions = {}
    for package in ("moelora-repro", "torch", "matplotlib", "numpy", "pandas",
                    "transformers", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def run_recipe(recipe, output_dir, recipe_path):
    output_dir = Path(output_dir).resolve()
    # Refuse even an empty existing directory: every run owns a new directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", ENTRYPOINTS[recipe["entrypoint"]],
               *parameter_args(recipe["parameters"]), "--out", str(output_dir)]
    (output_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "running", "started_at": utc_now(), "finished_at": None,
        "recipe_source": str(Path(recipe_path).resolve()), "command": command,
        "working_directory": str(Path.cwd()),
        "code_git": git_metadata(Path(__file__).resolve().parent),
        "recipe_git": git_metadata(Path(recipe_path).resolve().parent),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": package_versions(), "exit_code": None,
    }
    manifest_path = output_dir / "manifest.json"

    def save_manifest():
        temp = output_dir / "manifest.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp.replace(manifest_path)

    save_manifest()
    print(f"Run directory: {output_dir}", flush=True)
    process = None
    try:
        with (output_dir / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        manifest.update(status="completed" if return_code == 0 else "failed", exit_code=return_code)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error), exit_code=process.returncode if process is not None else None)
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        manifest["finished_at"] = utc_now()
        save_manifest()
    return return_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print the recipe without training or writing files")
    args = parser.parse_args(argv)
    try:
        recipe = load_recipe(args.config, args.overrides)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(recipe, indent=2))
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.out if args.out is not None else Path("outputs") / recipe["name"] / run_id
    try:
        return run_recipe(recipe, output, args.config)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output}; choose a new --out")


if __name__ == "__main__":
    raise SystemExit(main())
