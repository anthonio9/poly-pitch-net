from poly_pitch_net.models import FretNetCrepe
from torch import nn
import torch


def generate_dummy_batch(B, T, C, F, W):
    """
    Generate a dummy batch of size [B x T x C x F x W].

    Parameters
    ----------
    Input features for a batch of random values,
    B - batch size
    T - number of frames
    C - number of channels in features
    F - number of features (frequency bins)
    W - frame width of each sample

    Returns
    ----------
    batch : a PyTorch tensor of size [B x T x C x F x W] with random values.
    """
    return torch.rand(size=(B, T, C, F, W))


def test_dummy_batch():
    t_shape = generate_dummy_batch(30, 200, 6, 144, 1).shape
    assert list(t_shape) == [30, 200, 6, 144, 1]


def test_fretnet_crepe_forward_shape():
    freq_bins = 144
    hcqt_channels = 6
    batch_size = 30
    no_frames = 200
    no_strings = 6
    no_pitch_bins = 360

    net = FretNetCrepe(dim_in=freq_bins, in_channels=hcqt_channels)
    dummy_batch = generate_dummy_batch(B=batch_size,
                                       T=no_frames,
                                       C=hcqt_channels,
                                       F=freq_bins, W=1)

    output = net.forward(dummy_batch)

    expected_shape = [batch_size,
                      no_strings,
                      no_pitch_bins,
                      no_frames]

    assert list(output.shape) == expected_shape



