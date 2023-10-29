import poly_pitch_net as ppn

import amt_tools.tools
from amt_tools.features import HCQT
from guitar_transcription_continuous.datasets import GuitarSetPlus as GuitarSet
import torch
import random
import librosa


def run(gpu=None):
    root = ppn.tools.misc.get_project_root().parent

    # Build the path to GuitarSet
    gset_base_dir = os.path.join(str(root), '..', 'Datasets', 'GuitarSet')

    # Keep all cached data/features here
    gset_cache = os.path.join(str(root), '..', 'generated', 'data')
    gset_cache_train = os.path.join(gset_cache, 'train') # No extras
    gset_cache_val = os.path.join(gset_cache, 'val') # Includes extras

    random.seed(1234)
    k = random.randrange(ppn.GSET_NO_PLAYERS)

    # Allocate training/testing splits
    train_splits = GuitarSet.available_splits()
    val_splits = [train_splits.pop(k), train_splits.pop(k-1)]

    # Amount of semitones in each direction modeled for each note
    semitone_radius = 1.0

    # Flag to use rotarized pitch deviations for ground-truth
    rotarize_deviations = False # set to false in GuitarSet init anyway

    # Whether to perform data augmentation (pitch shifting) during training
    augment_data = False # set to false in GuitarSet init anyway

    # Flag to include an activation for silence in applicable output layers
    silence_activations = True

    # Whether to use cluster-based or ground-truth index-
    # based method for grouping notes and pitch contours
    use_cluster_grouping = True

    # Whether to use discrete targets derived from
    # pitch contours instead of notes for training
    use_adjusted_targets = True

    # Create an HCQT feature extraction module comprising
    # the first five harmonics and a sub-harmonic, where each
    # harmonic transform spans 4 octaves w/ 3 bins per semitone
    data_proc = HCQT(sample_rate=ppn.SAMPLE_RATE,
                     hop_length=ppn.HOPSIZE,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

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
                              batch_size=pnn.BATCH_SIZE,
                              shuffle=True,
                              num_workers=4 * int(augment_data),
                              drop_last=True)

    # Create a dataset corresponding to the validation partition
    gset_val = GuitarSet(base_dir=ppn.GSET_BASE_DIR,
                         splits=val_splits,
                         hop_length=ppn.HOPSIZE,
                         sample_rate=ppn.SAMPLE_RATE,
                         num_frames=None,
                         data_proc=data_proc,
                         profile=profile,
                         store_data=True,
                         save_loc=ppn.GSET_CACHE_VAL,
                         semitone_radius=semitone_radius,
                         rotarize_deviations=rotarize_deviations,
                         silence_activations=silence_activations,
                         use_cluster_grouping=use_cluster_grouping,
                         evaluation_extras=True)

    model = ppn.models.FretNetCrepe(
            dim_in=ppn.HCQT_DIM_IN,
            in_channels=ppn.HCQT_NO_HARMONICS,
            no_pitch_bins=ppn.PITCH_BINS
            )

    model.change_device()
    model.train()

    train(train_loader, gset_val, model, gpu=model.cuda())

def train(
        train_loader,
        val_dataset,
        model,
        gpu=None):
    pass



