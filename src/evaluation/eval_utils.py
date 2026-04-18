"""
Evaluation utilities — imported by notebooks/05_evaluation.ipynb
-----------------------------------------------------------------
Centralises checkpoint loading and model resolution so the notebook
stays clean and these helpers can be tested independently.
"""

from pathlib import Path
import torch
import torch.nn as nn


def load_state_dict_flexible(model: nn.Module,
                              ckpt_path: Path,
                              device: torch.device) -> nn.Module:
    """
    Load a model state dict that may be stored in one of two formats:

    1. Raw state dict (written by train_vqvae.py's final.pth):
           torch.save(model.state_dict(), path)

    2. Wrapped checkpoint dict (written by epoch checkpoints):
           torch.save({"epoch": N, "model": state_dict, ...}, path)

    Both are handled transparently.
    """
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    else:
        state = raw
    model.load_state_dict(state)
    return model


def find_latest_checkpoint(ckpt_dir: str | Path, model_name: str) -> Path | None:
    """
    Find the best available checkpoint for a model.
    Priority: final.pth > generator_final.pth > latest epoch_NNN.pth

    Args:
        ckpt_dir:   root checkpoint directory (Drive path)
        model_name: subdirectory name ('vqvae', 'swin_gan', etc.)

    Returns:
        Path to checkpoint, or None if nothing found.
    """
    # Bug fix: ckpt_dir may be a str (from config['data']['checkpoint_dir'])
    # Always wrap in Path() before joining.
    model_dir = Path(ckpt_dir) / model_name

    if not model_dir.exists():
        return None

    # Priority 1: final.pth (raw state dict, written at training end)
    for name in ("final.pth", "generator_final.pth"):
        p = model_dir / name
        if p.exists():
            return p

    # Priority 2: latest epoch checkpoint (wrapped dict format)
    epoch_ckpts = sorted(model_dir.glob("epoch_*.pth"))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    return None


def find_encoder_checkpoint(ckpt_dir: str | Path, model_name: str) -> Path | None:
    """Find the encoder (izi_f) checkpoint for a GAN model."""
    enc_path = Path(ckpt_dir) / model_name / "encoder_final.pth"
    return enc_path if enc_path.exists() else None
