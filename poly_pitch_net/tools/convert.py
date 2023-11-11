import poly_pitch_net as ppn
import torch


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    # return penn.CENTS_PER_BIN * bins
    return ppn.CENTS_PER_BIN * bins


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return ppn.OCTAVE * torch.log2(frequency / ppn.FMIN)


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return ppn.FMIN * 2 ** (cents / ppn.OCTAVE)


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = quantize_fn(cents / ppn.CENTS_PER_BIN).long()
    bins[bins < 0] = 0
    bins[bins >= ppn.PITCH_BINS] = ppn.PITCH_BINS - 1
    return bins


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)
