from pathlib import Path
import torch

import poly_pitch_net as ppn


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def cents(a, b):
    """Compute pitch difference in cents"""
    return ppn.OCTAVE * torch.log2(a / b)
