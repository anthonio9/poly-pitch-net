from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def cents(a, b):
    """Compute pitch difference in cents"""
    return penn.OCTAVE * torch.log2(a / b)
