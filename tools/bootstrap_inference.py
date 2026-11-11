"""Install the recorded P56 stack in a fresh Colab Python process; no model calls."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys


KNOWN_BASE_FINDING = "ipython 7.34.0 requires jedi, which is not installed."


def check_environment(evidence, actual):
    """Classify pip diagnostics strictly; readiness requires a successful probe."""
    record = dict(versions=actual, pip_check_returncode=None, pip_check_output="",
                  pip_check_stderr="", pip_check_classification="not_run", status="checking_dependencies",
                  resolver_plan="pip_plan.json", constraints="constraints.txt", model_calls=0)
    path = evidence / "environment.json"

    def save():
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    except Exception as exc:
        record.update(status="failed", pip_check_classification="execution_failed",
                      error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    record.update(pip_check_returncode=check.returncode, pip_check_output=check.stdout,
                  pip_check_stderr=check.stderr)
    finding = check.stdout.replace("\r\n", "\n").replace("\r", "\n").strip()
    if check.returncode == 0:
        classification = "passed"
    elif check.returncode == 1 and finding == KNOWN_BASE_FINDING and not check.stderr.strip():
        classification = "known_base_exception"
    else:
        record.update(status="failed", pip_check_classification="rejected")
        save()
        raise RuntimeError("Dependency check failed; see environment.json diagnostics")
    record.update(pip_check_classification=classification, status="checking_readiness")
    save()
    try:
        subprocess.run([sys.executable, "-c",
                        "import torch,torchvision,terratorch,lightning,rasterio; "
                        "assert torch.cuda.get_device_name(0)=='NVIDIA L4'; "
                        "assert torch.cuda.is_bf16_supported(); print('P56_DEPENDENCIES_READY')"], check=True)
    except Exception as exc:
        record.update(status="failed", readiness_probe="failed",
                      error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    record.update(status="ready" if classification == "passed" else "ready_with_known_base_exception",
                  readiness_probe="passed")
    save()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    expected = json.loads((source / "src/glacial_lake/inference_config.json").read_text())["versions"]
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    print("P56_BOOTSTRAP_TORCH", flush=True)
    subprocess.run([*base, "torch==" + expected["torch"], "torchvision==" + expected["torchvision"],
                    "--index-url", "https://download.pytorch.org/whl/cu130"], check=True)
    constraints = evidence / "constraints.txt"
    constraints.write_text("".join(k + "==" + v + "\n" for k, v in expected.items() if k != "terratorch"))
    install = [*base, "--constraint", str(constraints), str(source) + "[science]"]
    print("P56_BOOTSTRAP_RESOLVER", flush=True)
    plan = evidence / "pip_plan.json"
    subprocess.run([*install, "--dry-run", "--report", str(plan)], check=True)
    # Freeze every proposed dependency version before executing this resolution.
    proposed = json.loads(plan.read_text())["install"]
    with constraints.open("a") as stream:
        for item in proposed:
            name, version = item["metadata"]["name"], item["metadata"]["version"]
            if name.lower().replace("_", "-") not in {"glacial-lake", "terratorch", *expected}:
                stream.write(name + "==" + version + "\n")
    subprocess.run(install, check=True)
    actual = {name: metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError("Installed scientific versions differ")
    check_environment(evidence, actual)


if __name__ == "__main__":
    main()
