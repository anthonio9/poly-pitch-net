import torch
import random
import poly_pitch_net as ppn
from guitar_transcription_continuous.datasets import GuitarSetPlus as GuitarSet


def run(gpu=None):
    root = ppn.tools.misc.get_project_root().parent

    # Build the path to GuitarSet
    gset_base_dir = os.path.join(str(root), '..', 'Datasets', 'GuitarSet')

    # Keep all cached data/features here
    gset_cache = os.path.join(str(root), '..', 'generated', 'data')
    gset_cache_train = os.path.join(gset_cache, 'train') # No extras
    gset_cache_val = os.path.join(gset_cache, 'val') # Includes extras

    k = random.randint(ppn.NO_STRINGS)

    # Allocate training/testing splits
    train_splits = GuitarSet.available_splits()
    test_splits = [train_splits.pop(k)]
    val_splits = [train_splits.pop(k - 1)]

    # Create a dataset corresponding to the training partition
    gset_train = GuitarSet(base_dir=ppn.GSET_BASE_DIR,
                           splits=train_splits,
                           hop_length=ppn.HOPSIZE,
                           sample_rate=ppn.SAMPLE_RATE,
                           num_frames=ppn.NUM_FRAMES,
                           data_proc=data_proc,
                           profile=profile,
                           reset_data=(reset_data and k == 0),
                           save_loc=ppn.GSET_CACHE_TRAIN,
                           semitone_radius=semitone_radius,
                           rotarize_deviations=rotarize_deviations,
                           augment=augment_data,
                           silence_activations=silence_activations,
                           use_cluster_grouping=use_cluster_grouping,
                           use_adjusted_targets=use_adjusted_targets,
                           evaluation_extras=False)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=gset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4 * int(augment_data),
                              drop_last=True)


def train(
        gpu=None):
    pass



