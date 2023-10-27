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


@pytest.fixture(scope="session")
def base_config():
    # alternative solution with dotwiz package
    class config():
        pass

    config = config()

    config.freq_bins = 144
    config.hcqt_channels = 6
    config.batch_size = 30
    config.no_frames = 200
    config.no_strings = 6
    config.no_pitch_bins = 360

    return config


@pytest.fixture(scope="session")
def base_config_small():
    # alternative solution with dotwiz package
    class config():
        pass

    config = config()

    config.freq_bins = 5
    config.hcqt_channels = 2
    config.batch_size = 3
    config.no_frames = 5
    config.no_strings = 2
    config.no_pitch_bins = 10

    return config


@pytest.fixture(scope="session")
def fretnet_small(base_config_small):
    net = FretNetCrepe(dim_in=base_config_small.freq_bins,
                       in_channels=base_config_small.hcqt_channels,
                       no_strings=base_config_small.no_strings,
                       no_pitch_bins=base_config_small.no_pitch_bins)
    return net


def test_dummy_batch():
    t_shape = generate_dummy_batch(30, 200, 6, 144, 1).shape
    assert list(t_shape) == [30, 200, 6, 144, 1]


def test_fretnet_crepe_forward_shape(fretnet, base_config):

    dummy_batch = generate_dummy_batch(B=base_config.batch_size,
                                       T=base_config.no_frames,
                                       C=base_config.hcqt_channels,
                                       F=base_config.freq_bins, W=1)

    output = fretnet.forward(dummy_batch)[key_names.KEY_PITCH_LAYER]

    expected_shape = [base_config.batch_size,
                      base_config.no_strings,
                      base_config.no_pitch_bins,
                      base_config.no_frames]

    assert list(output.shape) == expected_shape


def gen_random_post_proc_tensors(base_config):
    torch.manual_seed(0)
    # vector with values, shape [B, C, T]
    vals = torch.randint(low=0, high=base_config.no_pitch_bins,
                         size=(base_config.batch_size,
                               base_config.no_strings,
                               base_config.no_frames))

    # prepare a 1-hot vector batch, shape [B, C, O, T]
    vals_1hot = nn.functional.one_hot(vals,
                                      num_classes=base_config.no_pitch_bins)
    vals_1hot = vals_1hot.reshape(base_config.batch_size,
                                  base_config.no_strings,
                                  base_config.no_pitch_bins, -1)

    # create the pitch_names array
    pitch_names = torch.arange(0, base_config.no_pitch_bins)
    pitch_names = pitch_names.expand(base_config.batch_size,
                                     base_config.no_strings,
                                     base_config.no_frames, -1)

    return vals, vals_1hot, pitch_names


def test_fretnet_post_proc_centers(fretnet, base_config):
    # vals is the centers tensor
    # vals_1hot is the input tensor
    vals, vals_1hot, pitch_names = gen_random_post_proc_tensors(base_config)

    input = {}
    input[key_names.KEY_PITCH_LAYER] = vals_1hot

    output = fretnet.post_proc(input, pitch_names=pitch_names)

    assert vals.shape == output[key_names.KEY_PITCH_CENTERS].shape
    assert torch.equal(vals, output[key_names.KEY_PITCH_CENTERS])


def run_fretnet_post_proc_pitch_weighted_average(fretnet, config):
    # vals is the centers tensor
    # vals_1hot is the input tensor
    vals, vals_1hot, pitch_names = gen_random_post_proc_tensors(config)

    input = {}
    input[key_names.KEY_PITCH_LAYER] = vals_1hot

    output = fretnet.post_proc(input, pitch_names=pitch_names)

    # expected output
    output_hat = torch.empty(size=vals.shape)

    for batch_no in range(output_hat.shape[0]):
        # shape [C, O, T]
        batch = vals_1hot[batch_no, :, :, :]

        # shape [C, T, O]
        batch = batch.reshape(config.no_strings,
                              config.no_frames,
                              config.no_pitch_bins)

        for chan in range(config.no_strings):
            # shape [T]
            centers = vals[batch_no, chan, :]

            # shape [T]
            min_centers = centers - 4
            min_centers = min_centers.where(min_centers >= 0, 0)

            # shape [T]
            max_centers = centers + 4
            max_centers = max_centers.where(
                    max_centers < config.no_pitch_bins,
                    config.no_pitch_bins)

            # iterate through time dimension
            for time in range(config.no_frames):
                t_batch = batch[chan, time, :]

                # partial weighted average
                # get values from the batch for the time frame
                p_wg_avg = t_batch[min_centers[time]:max_centers[time]]

                # multiply by pitch name values from range min:max cents
                p_wg_avg *= pitch_names[batch_no, chan, time,
                                        min_centers[time]:max_centers[time]]
                p_wg_avg = p_wg_avg.sum()

                output_hat[batch_no, chan, time] = p_wg_avg

    output_hat = output_hat.unsqueeze(-1)

    return output, output_hat

    assert output_hat.shape == output[key_names.KEY_PITCH_WG_AVG].shape
    assert torch.equal(output_hat, output[key_names.KEY_PITCH_WG_AVG])


def test_fretnet_post_proc_pitch_weighted_average_small(fretnet_small,
                                                        base_config_small):
    output, output_hat = \
            run_fretnet_post_proc_pitch_weighted_average(
                    fretnet_small, base_config_small)

    assert output_hat.shape == output[key_names.KEY_PITCH_WG_AVG].shape
    assert torch.equal(output_hat, output[key_names.KEY_PITCH_WG_AVG])


def test_fretnet_post_proc_pitch_weighted_average(fretnet, base_config):
    output, output_hat = \
            run_fretnet_post_proc_pitch_weighted_average(
                    fretnet, base_config)

    assert output_hat.shape == output[key_names.KEY_PITCH_WG_AVG].shape
    assert torch.equal(output_hat, output[key_names.KEY_PITCH_WG_AVG])
