import amt_tools.tools
from amt_tools.features import HCQT
import librosa
from poly_pitch_net.datasets.guitarset import GuitarSetPPN
import poly_pitch_net as ppn
import torch

import random


def loader(partition: str='train', seed: int=None):
    """
    Prepare a GuitarSetPPN data loader.
    """

    if seed is None:
        seed = ppn.RANDOM_SEED

    random.seed(seed)
    k = random.randrange(ppn.GSET_PLAYERS)
    splits = GuitarSetPPN.available_splits()
    val_splits = [splits.pop(k), splits.pop(k - 1)]

    if 'val' in partition:
        dataset_cache_path = ppn.GSET_CACHE_VAL
        dataset_splits = val_splits
        dataset_seed = seed + 1
    elif 'pytest' in partition:
        dataset_cache_path = ppn.GSET_CACHE_PYTEST
        dataset_seed = seed
        dataset_seed = [val_splits[0]]
    else:
        dataset_cache_path = ppn.GSET_CACHE_TRAIN
        dataset_seed = seed
        dataset_splits = splits

    # Create an HCQT feature extraction module comprising
    # the first five harmonics and a sub-harmonic, where each
    # harmonic transform spans 4 octaves w/ 3 bins per semitone
    data_proc = HCQT(sample_rate=ppn.GSET_SAMPLE_RATE,
                     hop_length=ppn.GSET_HOP_LEN,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    print(f"Preparing the dataset in {dataset_cache_path}")
    # Create a dataset
    gset = GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                           splits=dataset_splits,
                           num_frames=ppn.NUM_FRAMES,
                           data_proc=data_proc,
                           profile=profile,
                           reset_data=False, # set to true in the future trainings
                           save_data=True, # set to true in the future trainings
                           save_loc=dataset_cache_path,
                           seed=dataset_seed)

    # Create a PyTorch data loader for the dataset
    loader = torch.utils.data.DataLoader(dataset=gset,
                              batch_size=ppn.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)

    return loader
