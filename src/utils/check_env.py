"""Step 0 — verify the GPU node's environment before any real work.

Reports: GPU, torch/CUDA versions, installed vs missing packages, whether this node
can reach Hugging Face (model downloads) and PyPI (pip installs), free disk, and
whether the raw data is in place. Uses only the standard library + torch, so it
still runs (and reports) if other packages are missing.
"""
import os
# Must be set BEFORE importing torch / transformers.
# Inside a Slurm job, Slurm already restricts which GPU is visible, and the
# allocated GPU appears as index 0. So use "0" -- never a physical GPU number.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - refusing to run on CPU. "
                       "Did you submit via sbatch/srun with --gres=gpu:1?")
device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))

import argparse
import importlib
import json
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # .../Project_NLP


def reachable(host):
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": {},
    }

    # pip name -> import name
    for pip_name, mod in [("numpy", "numpy"), ("pyyaml", "yaml"), ("pandas", "pandas"),
                          ("scikit-learn", "sklearn"), ("transformers", "transformers"),
                          ("sentence-transformers", "sentence_transformers"),
                          ("huggingface-hub", "huggingface_hub"),
                          ("streamlit", "streamlit"), ("altair", "altair")]:
        try:
            report["packages"][pip_name] = getattr(importlib.import_module(mod), "__version__", "installed")
        except ImportError:
            report["packages"][pip_name] = "MISSING"

    report["internet_huggingface"] = reachable("huggingface.co")
    report["internet_pypi"] = reachable("pypi.org")
    report["HF_HOME"] = os.environ.get("HF_HOME", "(not set - default ~/.cache/huggingface)")
    report["disk_free_GB"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)

    try:
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
        report["raw_data_present"] = {k: (ROOT / v).exists()
                                      for k, v in cfg["paths"].items() if k.startswith("raw_")}
    except ImportError:
        report["raw_data_present"] = "not checked (pyyaml missing)"

    print("\n========== ENVIRONMENT REPORT ==========")
    for k, v in report.items():
        print(f"{k}: {v}")

    missing = [k for k, v in report["packages"].items() if v == "MISSING"]
    print("\n========== SUMMARY ==========")
    print("Missing packages:", missing if missing else "none")
    print("Can install packages from this node (PyPI):", report["internet_pypi"])
    print("Can download models from this node (Hugging Face):", report["internet_huggingface"])

    out = ROOT / "outputs" / "results" / "env_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()