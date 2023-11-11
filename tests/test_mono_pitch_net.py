import poly_pitch_net as ppn
import torch


def test_mpn_forward():

    model = ppn.models.MonoPitchNet1D(
            dim_in=ppn.HCQT_DIM_IN,
            no_pitch_bins=ppn.PITCH_BINS
            )

    batch_size = 1
    freq = ppn.HCQT_DIM_IN
    no_frames = 20
    no_pitch_bins = ppn.PITCH_BINS

    features = torch.rand(batch_size, no_frames, freq)

    # try to change for cuda:0
    model.change_device(0)

    # forward pass
    features.to(model.device)

    output = model(features)

    expected_shape = [batch_size,
                      no_pitch_bins,
                      no_frames]

    assert list(output.shape) == expected_shape
