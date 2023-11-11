import poly_pitch_net as ppn
import pytest
import torch


@pytest.fixture(scope="session")
def cfg_small():
    # alternative solution with dotwiz package
    class config():
        pass

    config = config()

    config.freq_bins = 144
    config.batch_size = 1
    config.no_frames = 20
    config.no_pitch_bins = 360

    return config


def mono_pitch_net1d(config):
    model = ppn.models.MonoPitchNet1D(
            dim_in=config.freq_bins,
            no_pitch_bins=config.no_pitch_bins
            )

    return model


def test_mpn_forward(cfg_small):
    cfg = cfg_small

    model = ppn.models.MonoPitchNet1D(
            dim_in=cfg.freq_bins,
            no_pitch_bins=cfg.no_pitch_bins
            )

    features = torch.rand(cfg.batch_size, cfg.no_frames, cfg.freq_bins)

    # try to change for cuda:0
    model.change_device(0)

    # forward pass
    features.to(model.device)

    output = model(features)

    # [B, O, T]
    expected_shape = [cfg.batch_size,
                      cfg.no_pitch_bins,
                      cfg.no_frames]

    print(f"output.shape: {output[ppn.KEY_PITCH_LOGITS].shape}")

    assert list(output[ppn.KEY_PITCH_LOGITS].shape) == expected_shape


def test_mpn_loss(cfg_small):
    cfg = cfg_small

    # prepare data
    logits = torch.rand(cfg.batch_size, cfg.no_pitch_bins, cfg.no_frames)
    pitch_array = torch.arange(120, 1200, 30)[:cfg.no_frames]

    ppn.train.mono_pitch_loss(logits, pitch_array)
