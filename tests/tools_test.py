import amt_tools.tools
import poly_pitch_net as ppn
from poly_pitch_net.datasets import GuitarSetPPN
import torch


def test_get_project_root():
    root = ppn.tools.misc.get_project_root()

    assert root.stem == 'poly_pitch_net'


def test_tablature_to_pitch():
    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    # create a train_loader
    gset_train = GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                           splits=[GuitarSetPPN.available_splits().pop(0)],
                           hop_length=ppn.HOPSIZE,
                           sample_rate=ppn.SAMPLE_RATE,
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
    batch = ppn.tools.convert.tablature_rel_to_pitch(batch, profile)

