import poly_pitch_net as ppn
import poly_pitch_net.datasets.guitarset as guitarset

import amt_tools.tools
from amt_tools.features import HCQT
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import random
import librosa
import torchutil


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
    data_proc = HCQT(sample_rate=guitarset.GSET_SAMPLE_RATE,
                     hop_length=guitarset.GSET_HOP_LEN,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    print(f"Preparing the trian set in {ppn.GSET_CACHE_TRAIN}")
    # Create a dataset corresponding to the training partition
    gset_train = guitarset.GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                           splits=train_splits,
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
                              drop_last=True)

    print(f"Preparing the validation set in {ppn.GSET_CACHE_VAL}")
    # Create a dataset corresponding to the validation partition
    gset_val = guitarset.GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                         splits=val_splits,
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

    model.change_device(device=0)
    model.train()

    print("Starting the training")
    train(train_loader, gset_val, model, model_dir)

def train(
        train_loader,
        val_loader,
        model,
        log_dir):

    # Initialize a writer to log any reported results
    writer = SummaryWriter(log_dir)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ppn.LEARNING_RATE)

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    step, epoch = 0, 0

    for iter in tqdm(range(ppn.STEPS)):
        # Loop through the dataset
        for batch in train_loader:

            breakpoint()
            # Unpack batch
            features = batch[key_names.KEY_FEATURES]
            pitch_array = batch[key_names.KEY_PITCH_ARRAY]

            with torch.autocast(model.device.type):

                # Forward pass
                logits = model(features.to(model.device))

                # Compute losses
                losses = ppn.train.loss(logits, pitch_array.to(model.device))

            # Zero the accumulated gradients
            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # log the loss
            writer.add_scalar('train_loss ' + ppn.LOSS_BCE, losses)

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            step += 1

        epoch += 1

    # Save final model
    torchutil.checkpoint.save(
        log_dir / f'{step:08d}.pt',
        model,
        optimizer,
        step=step,
        epoch=epoch)


def evaluate(
        loader: torch.utils.data.DataLoader,
        model: ppn.models.FretNetCrepe,
        writer: SummaryWriter):
    """
    Perform model evaluation.
    """

    model.eval()

    for batch in loader:
        features = batch[key_names.KEY_FEATURES]
        pitch_array = batch[key_names.KEY_PITCH_ARRAY]

        # set the pitch names to something
        model.post_proc(model(features), pitch_names=?)
