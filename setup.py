#!/usr/bin/env python3
"""
Helper script to mirror the README setup steps:
  1. Create directory structure for repository, weights, environment, and designs.
  2. Optionally download published model weights and the SE3nv Apptainer image.

Example:
  python setup.py --base-dir ~/git/RFDpoly_paper_version --download-all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple
import urllib.request
import shutil


RNA_WEIGHTS_URL = "https://files.ipd.uw.edu/pub/2025_RFDpoly/train_session2024-06-27_1719522052_BFF_7.00.pt"
MULTI_WEIGHTS_URL = "https://files.ipd.uw.edu/pub/2025_RFDpoly/train_session2024-07-08_1720455712_BFF_3.00.pt"
APPTAINER_URL = "https://files.ipd.uw.edu/pub/2025_RFDpoly/SE3nv.sif"


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def download(url: str, destination: Path, overwrite: bool) -> Tuple[bool, str]:
    if destination.exists() and not overwrite:
        return False, f"Skipped existing file: {destination}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    return True, f"Downloaded {url} -> {destination}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up local directories and optional downloads.")
    parser.add_argument("--base-dir", default="~/git/RFDpoly_paper_version",
                        help="Root directory used in the README examples. Defaults to %(default)s")
    parser.add_argument("--repo-dir", default="~/git/RFDpoly_paper_version/polydiff",
                        help="Existing clone location for this repository.")
    parser.add_argument("--weights-dir", default="~/git/RFDpoly_paper_version/weights",
                        help="Location to store downloaded model weights.")
    parser.add_argument("--env-dir", default="~/git/RFDpoly_paper_version/exec",
                        help="Directory for Apptainer image or environment files.")
    parser.add_argument("--design-dir", default="~/git/RFDpoly_paper_version/design_jobs",
                        help="Directory for running design jobs.")
    parser.add_argument("--download-rna-weights", action="store_true",
                        help="Download the RNA-only checkpoint.")
    parser.add_argument("--download-multi-weights", action="store_true",
                        help="Download the multi-polymer checkpoint.")
    parser.add_argument("--download-apptainer", action="store_true",
                        help="Download the SE3nv Apptainer image.")
    parser.add_argument("--download-all", action="store_true",
                        help="Enable all download flags.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite files if they already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base_dir = expand(args.base_dir)
    repo_dir = expand(args.repo_dir)
    weights_dir = expand(args.weights_dir)
    env_dir = expand(args.env_dir)
    design_dir = expand(args.design_dir)

    ensure_directories([base_dir, weights_dir, env_dir, design_dir])

    if not repo_dir.exists():
        print(f"[warn] Repository directory '{repo_dir}' not found. Clone the repo before running setup.py.")

    download_flags = {
        "rna": args.download_rna_weights or args.download_all,
        "multi": args.download_multi_weights or args.download_all,
        "apptainer": args.download_apptainer or args.download_all,
    }

    actions = []
    try:
        if download_flags["rna"]:
            dest = weights_dir / Path(RNA_WEIGHTS_URL).name
            changed, message = download(RNA_WEIGHTS_URL, dest, args.overwrite)
            actions.append(message)
            if changed:
                os.chmod(dest, 0o644)
        if download_flags["multi"]:
            dest = weights_dir / Path(MULTI_WEIGHTS_URL).name
            changed, message = download(MULTI_WEIGHTS_URL, dest, args.overwrite)
            actions.append(message)
            if changed:
                os.chmod(dest, 0o644)
        if download_flags["apptainer"]:
            dest = env_dir / Path(APPTAINER_URL).name
            changed, message = download(APPTAINER_URL, dest, args.overwrite)
            actions.append(message)
            if changed:
                os.chmod(dest, 0o755)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print("Setup summary:")
    print(f"  Base directory:     {base_dir}")
    print(f"  Repository clone:   {repo_dir}")
    print(f"  Weights directory:  {weights_dir}")
    print(f"  Environment directory: {env_dir}")
    print(f"  Design jobs dir:    {design_dir}")
    print()
    print("Suggested environment variables:")
    print(f"  export RFDPOLY_DIR='{base_dir}'")
    print(f"  export WEIGHTS_DIR='{weights_dir}'")
    print(f"  export ENV_DIR='{env_dir}'")
    print(f"  export DESIGN_DIR='{design_dir}'")
    if download_flags["multi"]:
        print(f"  export MODEL_WEIGHTS_PATH='{weights_dir / Path(MULTI_WEIGHTS_URL).name}'")
    elif download_flags["rna"]:
        print(f"  export MODEL_WEIGHTS_PATH='{weights_dir / Path(RNA_WEIGHTS_URL).name}'")
    if download_flags["apptainer"]:
        print(f"  export APPTAINER_PATH='{env_dir / Path(APPTAINER_URL).name}'")

    if actions:
        print("\nDownloads:")
        for note in actions:
            print(f"  - {note}")
    else:
        print("\nNo downloads requested.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
