#!/usr/bin/env python3
"""Download VisDrone-DET dataset.

Downloads the VisDrone-DET train/val/test splits from the official mirror.
Extracts images into data/visdrone/{train,val,test}/images/.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URLS = {
    "train": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1/VisDrone2019-DET-train.zip",
    "val": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1/VisDrone2019-DET-val.zip",
    "test": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1/VisDrone2019-DET-test-dev.zip",
}


def download_and_extract(url: str, dest: Path) -> None:
    """Download a URL and extract to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    archive_name = url.split("/")[-1]
    archive_path = dest / archive_name

    if archive_path.exists():
        print(f"  Archive already exists: {archive_path}")
    else:
        print(f"  Downloading {url}...")
        urlretrieve(url, archive_path)

    print(f"  Extracting to {dest}...")
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest)
    elif archive_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest)

    print(f"  Done: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download VisDrone-DET dataset")
    parser.add_argument("--root", type=str, default="data/visdrone", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to download")
    args = parser.parse_args()

    root = Path(args.root)
    for split in args.splits:
        if split not in URLS:
            print(f"Unknown split: {split}. Available: {list(URLS.keys())}")
            sys.exit(1)
        print(f"\n=== {split} ===")
        download_and_extract(URLS[split], root)

    print(f"\nAll done. Dataset at: {root}")


if __name__ == "__main__":
    main()
