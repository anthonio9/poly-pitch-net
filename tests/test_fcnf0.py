import poly_pitch_net as ppn

import amt_tools
from amt_tools.features import HCQT, STFT
import librosa
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def fncf0_cfg():
    batch_size = 48
    sr = 11025
    frame_size = 1024
    seq_length = 256 // 4

    return batch_size, sr, frame_size, seq_length


@pytest.fixture(scope="session", autouse=True)
def get_dataset(fncf0_cfg):
    batch_size, sample_rate, frame_size, seq_length = fncf0_cfg
    dataset_cache_path, dataset_seed, dataset_splits = ppn.datasets.get_dataset_path_seed_splits('pytest')

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    data_proc = STFT(hop_length=ppn.GSET_HOP_LEN // 4,
                     sample_rate=sample_rate)

    gset = ppn.datasets.GuitarSetPPN(
            hop_length=ppn.GSET_HOP_LEN // 4,
            base_dir=ppn.GSET_BASE_DIR,
            sample_rate=sample_rate,
            splits=dataset_splits,
            num_frames=frame_size // seq_length,
            data_proc=data_proc,
            profile=profile,
            reset_data=False,  # set to true in the future trainings
            save_data=True,  # set to true in the future trainings
            save_loc=dataset_cache_path / 'fcnf0',
            seed=dataset_seed)

    # Create a PyTorch data loader for the dataset
    loader = torch.utils.data.DataLoader(
            dataset=gset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)

    return loader


def test_fcnf0_dataset(fncf0_cfg, get_dataset):
    batch_size, sample_rate, frame_size, seq_length = fncf0_cfg
    loader = get_dataset

    loader = iter(loader)
    batch = next(loader)

    assert list(batch[ppn.KEY_AUDIO].shape) == [batch_size, frame_size]


def test_fcnf0_pre_proc(fncf0_cfg, get_dataset):
    batch_size, sample_rate, frame_size, seq_length = fncf0_cfg

    loader = get_dataset
    loader = iter(loader)
    batch = next(loader)

    model = ppn.models.FCNF0()
    output = model.pre_proc(batch)

    assert list(output[ppn.KEY_AUDIO].shape) == [batch_size, 1, 1024]
    assert list(output[ppn.KEY_PITCH_ARRAY].shape) == [batch_size, 1]


def test_fcnf0_forward(fncf0_cfg, get_dataset):
    batch_size, sample_rate, frame_size, seq_length = fncf0_cfg

    loader = get_dataset
    loader = iter(loader)
    batch = next(loader)

    model = ppn.models.FCNF0()
    output = model.forward(batch)

    assert list(output[ppn.KEY_PITCH_LOGITS].shape) == [batch_size, ppn.PITCH_BINS, 1]
