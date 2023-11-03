import amt_tools.tools
import poly_pitch_net as ppn
from poly_pitch_net.datasets import guitarset
from poly_pitch_net.tools import key_names
import torch


def test_get_project_root():
    root = ppn.tools.misc.get_project_root()

    assert root.stem == 'poly_pitch_net'


def test_guitarset_batch():
    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    # create a train_loader
    gset_train = guitarset.GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                           splits=[guitarset.GuitarSetPPN.available_splits().pop(0)],
                           num_frames=ppn.NUM_FRAMES,
                           profile=profile,
                           reset_data=False, # set to true in the future trainings
                           save_data=True, # set to true in the future trainings
                           save_loc=ppn.GSET_CACHE_PYTEST,
                           seed=ppn.RANDOM_SEED)

    # Create a PyTorch data loader for the dataset
    train_loader = torch.utils.data.DataLoader(dataset=gset_train,
                              batch_size=ppn.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)

    train_loader = iter(train_loader)
    batch = next(train_loader)

    pitchlist_shape = (ppn.BATCH_SIZE, key_names.GSET_PLAYERS, ppn.NUM_FRAMES)
    assert batch[key_names.KEY_PITCH_ARRAY].shape == pitchlist_shape
