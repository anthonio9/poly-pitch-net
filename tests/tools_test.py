import amt_tools.tools
import poly_pitch_net as ppn
from poly_pitch_net.datasets.guitarset import GuitarSetPPN
from amt_tools.features import HCQT
import librosa
import os
import random
import torch
from tqdm import tqdm
import zlib


def test_get_project_root():
    root = ppn.tools.misc.get_project_root()

    assert root.stem == 'poly_pitch_net'


def test_guitarset_batch():
    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    # Create an HCQT feature extraction module comprising
    # the first five harmonics and a sub-harmonic, where each
    # harmonic transform spans 4 octaves w/ 3 bins per semitone
    data_proc = HCQT(sample_rate=ppn.GSET_SAMPLE_RATE,
                     hop_length=ppn.GSET_HOP_LEN,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    # create a train_loader
    gset_train = GuitarSetPPN(
            base_dir=ppn.GSET_BASE_DIR,
            splits=[GuitarSetPPN.available_splits().pop(0)],
            num_frames=ppn.NUM_FRAMES,
            profile=profile,
            data_proc=data_proc,
            reset_data=False, # set to true in the future trainings
            save_data=True, # set to true in the future trainings
            save_loc=ppn.GSET_CACHE_PYTEST,
            seed=ppn.RANDOM_SEED)

    # Create a PyTorch data loader for the dataset
    train_loader = torch.utils.data.DataLoader(dataset=gset_train,
                              batch_size=ppn.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)

    pitchlist_shape = (ppn.BATCH_SIZE, ppn.GSET_PLAYERS, ppn.NUM_FRAMES)

    progress = tqdm(range(len(train_loader.dataset) // ppn.BATCH_SIZE))
    for batch in train_loader:
        assert batch[ppn.KEY_PITCH_ARRAY].shape == pitchlist_shape
        progress.update()


def test_guitarset_batch_train():
    random.seed(ppn.RANDOM_SEED)
    torch.manual_seed(ppn.RANDOM_SEED)
    k = random.randrange(ppn.GSET_PLAYERS)

    # Allocate training/testing splits
    train_splits = GuitarSetPPN.available_splits()
    val_splits = [train_splits.pop(k), train_splits.pop(k-1)]

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    # Create an HCQT feature extraction module comprising
    # the first five harmonics and a sub-harmonic, where each
    # harmonic transform spans 4 octaves w/ 3 bins per semitone
    data_proc = HCQT(sample_rate=ppn.GSET_SAMPLE_RATE,
                     hop_length=ppn.GSET_HOP_LEN,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    # create a train_loader
    gset_train = GuitarSetPPN(
            base_dir=ppn.GSET_BASE_DIR,
            splits=train_splits,
            num_frames=ppn.NUM_FRAMES,
            profile=profile,
            data_proc=data_proc,
            reset_data=False, # set to true in the future trainings
            save_data=True, # set to true in the future trainings
            save_loc=ppn.GSET_CACHE_TRAIN,
            seed=ppn.RANDOM_SEED)

    # Create a PyTorch data loader for the dataset
    train_loader = torch.utils.data.DataLoader(dataset=gset_train,
                              batch_size=ppn.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)

    pitchlist_shape = (ppn.BATCH_SIZE, ppn.GSET_PLAYERS, ppn.NUM_FRAMES)

    
    progress = tqdm(range(len(train_loader.dataset) // ppn.BATCH_SIZE))
    for batch in train_loader:
        assert batch[ppn.KEY_PITCH_ARRAY].shape == pitchlist_shape
        progress.update()


def test_guitarset_ungzip_train():
    gtrain_path = ppn.GSET_CACHE_TRAIN / 'GuitarSetPPN' / 'HCQT'

    failed = 0
    succed = 0

    for file in os.listdir(gtrain_path):
        try:
            batch = amt_tools.tools.load_dict_npz(gtrain_path / file)
        except zlib.error as err:
            print(f"{failed}: file {gtrain_path / file} failed to ungzip \n {err}") 
            failed += 1
        else: 
            assert ppn.KEY_FEATURES in batch
            succed += 1
    
    print(f"ungzip success with {succed} files!")

