import poly_pitch_net as ppn
import torch


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    # return penn.CENTS_PER_BIN * bins
    return ppn.CENTS_PER_BIN * bins


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return ppn.OCTAVE * torch.log2(frequency / ppn.FMIN)
