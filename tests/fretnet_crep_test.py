import amt_tools.tools as tools
from poly_pitch_net.models import FretNetCrepe
from poly_pitch_net.tools import key_names

import pytest
import torch
from torch import nn


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


@pytest.fixture(scope="session")
def fretnet():
    freq_bins = 144
    hcqt_channels = 6

    net = FretNetCrepe(dim_in=freq_bins, in_channels=hcqt_channels)
    return net


def test_dummy_batch():
    t_shape = generate_dummy_batch(30, 200, 6, 144, 1).shape
    assert list(t_shape) == [30, 200, 6, 144, 1]


def test_fretnet_crepe_forward_shape(fretnet):
    freq_bins = 144
    hcqt_channels = 6
    batch_size = 30
    no_frames = 200
    no_strings = 6
    no_pitch_bins = 360

    dummy_batch = generate_dummy_batch(B=batch_size,
                                       T=no_frames,
                                       C=hcqt_channels,
                                       F=freq_bins, W=1)

    output = fretnet.forward(dummy_batch)[key_names.KEY_PITCH_LAYER]

    expected_shape = [batch_size,
                      no_strings,
                      no_pitch_bins,
                      no_frames]

    assert list(output.shape) == expected_shape


def test_fretnet_post_proc(fretnet):
    batch_size = 30
    no_frames = 200
    no_strings = 6
    no_pitch_bins = 360

    # vector with values, shape [B, C, T]
    vals = torch.randint(low=0, high=no_pitch_bins,
                         size=(batch_size, no_strings, no_frames))

    # prepare a 1-hot vector batch, shape [B, C, O, T]
    vals_1hot = nn.functional.one_hot(vals, num_classes=no_pitch_bins)
    vals_1hot = vals_1hot.reshape(batch_size, no_strings, no_pitch_bins, -1)

    input = {}
    input[key_names.KEY_PITCH_LAYER] = vals_1hot

    # create the pitch_names array
    pitch_names = torch.arange(0, no_pitch_bins)
    pitch_names = pitch_names.expand(batch_size, no_strings,
                                     no_frames, -1)

    output = fretnet.post_proc(input, pitch_names=pitch_names)

    assert vals.shape == output[key_names.KEY_PITCH_CENTERS].shape
    assert torch.equal(vals, output[key_names.KEY_PITCH_CENTERS])
