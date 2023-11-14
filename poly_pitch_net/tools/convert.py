import poly_pitch_net as ppn
import torch


NO_PITCH_BIN = -1000000


def bins_to_cents(bins, register_silence=False):
    """Converts pitch bins to cents"""
    # return penn.CENTS_PER_BIN * bins
    cents = ppn.CENTS_PER_BIN * bins

    if register_silence:
        cents[bins == ppn.PITCH_BINS] = NO_PITCH_BIN

    return cents


def frequency_to_cents(frequency, register_silence=False):
    """Convert frequency in Hz to cents"""
    cents = ppn.OCTAVE * torch.log2(frequency / ppn.FMIN)

    if register_silence:
        cents[frequency == 0] = NO_PITCH_BIN

    return cents


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return ppn.FMIN * 2 ** (cents / ppn.OCTAVE)


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = quantize_fn(cents / ppn.CENTS_PER_BIN).long()
    bins[bins < 0] = 0
    bins[bins >= ppn.PITCH_BINS] = ppn.PITCH_BINS - 1
    return bins


def frequency_to_bins(frequency, quantize_fn=torch.floor, register_silence=False):
    """Convert frequency in Hz to pitch bins"""
    bins = cents_to_bins(frequency_to_cents(frequency), quantize_fn)

    if register_silence:
        bins[frequency == 0] = ppn.PITCH_BINS

    return bins


def bins_to_frequency(bins, register_silence=False):
    frequency = cents_to_frequency(bins_to_cents(bins))

    if register_silence:
        # set the last bins class to 0, this means silence
        frequency[bins == ppn.PITCH_BINS] = 0

    return frequency
