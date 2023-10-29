import poly_pitch_net as ppn
from guitar_transcription_continuous.datasets import GuitarSetPlus as GuitarSet
import torch


def test_get_project_root():
    root = ppn.tools.misc.get_project_root()

    assert root.stem == 'poly_pitch_net'


def test_tablature_to_pitch():
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

    # create a train_loader
    gset_train = GuitarSet(base_dir=ppn.GSET_BASE_DIR,
                           splits=GuitarSet.available_splits(),
                           hop_length=ppn.HOPSIZE,
                           sample_rate=ppn.SAMPLE_RATE,
                           num_frames=ppn.NUM_FRAMES,
                           reset_data=False, # set to true in the future trainings
                           save_data=True, # set to true in the future trainings
                           save_loc=ppn.GSET_CACHE_PYTEST,
                           semitone_radius=semitone_radius,
                           rotarize_deviations=rotarize_deviations,
                           augment=augment_data,
                           silence_activations=silence_activations,
                           use_cluster_grouping=use_cluster_grouping,
                           use_adjusted_targets=use_adjusted_targets,
                           evaluation_extras=True,
                           seed=ppn.RANDOM_SEED)

    # Create a PyTorch data loader for the dataset
    train_loader = torch.utils.data.DataLoader(dataset=gset_train,
                              batch_size=ppn.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)

    train_loader = iter(train_loader)
    batch = next(train_loader)
    batch = ppn.tools.convert.tablature_rel_to_pitch(batch)

