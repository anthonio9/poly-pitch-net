import poly_pitch_net as ppn
import poly_pitch_net.datasets.guitarset as guitarset

import amt_tools.tools
from amt_tools.features import HCQT
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import random
import librosa


def run():
    EX_NAME = '_'.join([ppn.models.FretNetCrepe.model_name(),
                        guitarset.GuitarSetPPN.dataset_name(),
                        HCQT.features_name()])

    # Create the root directory for the experiment files
    experiment_dir = ppn.tools.misc.get_project_root().parent / '..' / 'generated' / 'experiments' / EX_NAME

    # Create a log directory for the training experiment
    model_dir = experiment_dir / 'models'

    random.seed(ppn.RANDOM_SEED)
    torch.manual_seed(ppn.RANDOM_SEED)
    k = random.randrange(ppn.GSET_NO_PLAYERS)

    # Allocate training/testing splits
    train_splits = guitarset.GuitarSet.available_splits()
    val_splits = [train_splits.pop(k), train_splits.pop(k-1)]

    # Create an HCQT feature extraction module comprising
    # the first five harmonics and a sub-harmonic, where each
    # harmonic transform spans 4 octaves w/ 3 bins per semitone
    data_proc = HCQT(sample_rate=ppn.SAMPLE_RATE,
                     hop_length=guitarset.GSET_HOP_LEN,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    print(f"Preparing the trian set in {ppn.GSET_CACHE_TRAIN}")
    # Create a dataset corresponding to the training partition
    gset_train = guitarset.GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                           splits=train_splits,
                           sample_rate=ppn.SAMPLE_RATE,
                           num_frames=ppn.NUM_FRAMES,
                           data_proc=data_proc,
                           profile=profile,
                           reset_data=False, # set to true in the future trainings
                           save_data=True, # set to true in the future trainings
                           save_loc=ppn.GSET_CACHE_TRAIN,
                           seed=ppn.RANDOM_SEED)

    # Create a PyTorch data loader for the dataset
    train_loader = torch.utils.data.DataLoader(dataset=gset_train,
                              batch_size=ppn.BATCH_SIZE,
                              shuffle=True,
                              num_workers=4 * int(augment_data),
                              drop_last=True)

    print(f"Preparing the validation set in {ppn.GSET_CACHE_VAL}")
    # Create a dataset corresponding to the validation partition
    gset_val = guitarset.GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                         splits=val_splits,
                         sample_rate=ppn.SAMPLE_RATE,
                         num_frames=None,
                         data_proc=data_proc,
                         profile=profile,
                         store_data=True,
                         save_loc=ppn.GSET_CACHE_VAL,
                         seed=ppn.RANDOM_SEED + 1)

    model = ppn.models.FretNetCrepe(
            dim_in=ppn.HCQT_DIM_IN,
            in_channels=ppn.HCQT_NO_HARMONICS,
            no_pitch_bins=ppn.PITCH_BINS
            )

    model.change_device()
    model.train()

    print("Starting the training")
    train(train_loader, gset_val, model, model_dir)

def train(
        train_loader,
        val_dataset,
        model,
        log_dir):

    # Initialize a writer to log any reported results
    writer = SummaryWriter(log_dir)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ppn.LEARNING_RATE)

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    for iter in tqdm(range(ppn.STEPS)):
        # Loop through the dataset
        for batch in train_loader:
            # Zero the accumulated gradients
            optimizer.zero_grad()

            breakpoint()
            # Unpack batch
            features = batch[ppn.KEY_FEATURES]
            tablature = batch[ppn.KEY_TABLATURE] # in hertz
            tablature_rel = batch[ppn.KEY_TABLATURE_REL]

            # have to convert TABLATURE and TABLATURE_REL into KEY_PITCH

            with torch.autocast(model.device.type):

                # Forward pass
                logits = model(audio.to(model.device))

                # Compute losses
                losses = ppn.train.loss(logits, bins.to(model.device))

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()



def evaluate():
    pass



