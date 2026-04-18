"""
IXI Dataset Downloader
-----------------------
Downloads T1 and T2 NIfTI files from the IXI dataset to a target directory.
Run this once in Colab notebook 01 (data_preparation).

Usage:
    python download_ixi.py --out_dir /content/drive/MyDrive/atml/data/ixi

IXI Dataset: http://brain-development.org/ixi-dataset/
License: CC BY-SA 3.0
~600 healthy volunteer brain MRIs from 3 London hospitals.
"""

import os
import argparse
import subprocess
from pathlib import Path

# Direct download URLs from the IXI project website
IXI_URLS = {
    "T1": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
    "T2": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
}


def download_and_extract(url: str, out_dir: Path, modality: str):
    """Download a tar archive and extract it."""
    tar_path = out_dir / f"IXI-{modality}.tar"

    if (out_dir / modality).exists() and len(list((out_dir / modality).glob("*.nii.gz"))) > 100:
        print(f"[{modality}] Already downloaded ({len(list((out_dir / modality).glob('*.nii.gz')))} files). Skipping.")
        return

    (out_dir / modality).mkdir(parents=True, exist_ok=True)

    print(f"[{modality}] Downloading from {url} ...")
    print(f"  This is ~2–3 GB. Expected time: 5–15 min depending on Colab network.")

    # wget with progress
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(tar_path), url],
        check=True
    )

    print(f"[{modality}] Extracting ...")
    subprocess.run(
        ["tar", "-xf", str(tar_path), "-C", str(out_dir / modality)],
        check=True
    )

    tar_path.unlink()  # Remove tar after extraction
    n_files = len(list((out_dir / modality).glob("*.nii.gz")))
    print(f"[{modality}] Done. {n_files} scans saved to {out_dir / modality}")


def verify_paired(out_dir: Path):
    """
    IXI naming convention: IXI012-HH-1211-T1.nii.gz / IXI012-HH-1211-T2.nii.gz
    Find subjects that have BOTH T1 and T2.
    """
    t1_ids = {f.name.replace("-T1.nii.gz", "") for f in (out_dir / "T1").glob("*.nii.gz")}
    t2_ids = {f.name.replace("-T2.nii.gz", "") for f in (out_dir / "T2").glob("*.nii.gz")}
    paired = sorted(t1_ids & t2_ids)
    print(f"\nPaired T1+T2 subjects available: {len(paired)}")
    print(f"  T1 only: {len(t1_ids - t2_ids)}")
    print(f"  T2 only: {len(t2_ids - t1_ids)}")

    # Save paired subject list for reproducibility
    paired_file = out_dir / "paired_subjects.txt"
    paired_file.write_text("\n".join(paired))
    print(f"  Paired subject list saved to: {paired_file}")
    return paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IXI T1+T2 dataset")
    parser.add_argument("--out_dir", type=str, required=True, help="Target directory (e.g. Drive path)")
    parser.add_argument("--modalities", nargs="+", default=["T1", "T2"], choices=["T1", "T2"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for modality in args.modalities:
        download_and_extract(IXI_URLS[modality], out_dir, modality)

    verify_paired(out_dir)
    print("\n✅ IXI download complete. Ready for preprocessing.")
