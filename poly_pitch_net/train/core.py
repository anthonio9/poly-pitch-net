import poly_pitch_net as ppn
from poly_pitch_net.datasets.guitarset import GuitarSetPPN
from poly_pitch_net.models import FretNetCrepe
import amt_tools.tools
from amt_tools.features import HCQT

from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import random
import librosa
import torchutil


def run():
    EX_NAME = '_'.join([FretNetCrepe.model_name(),
                        GuitarSetPPN.dataset_name(),
                        HCQT.features_name()])

    # Create the root directory for the experiment files
    experiment_dir = ppn.tools.misc.get_project_root().parent / '..' / 'generated' / 'experiments' / EX_NAME

    # Create a log directory for the training experiment
    model_dir = experiment_dir / 'models'

    random.seed(ppn.RANDOM_SEED)
    torch.manual_seed(ppn.RANDOM_SEED)
    k = random.randrange(ppn.GSET_PLAYERS)

    # Allocate training/testing splits
    train_splits = GuitarSetPPN.available_splits()
    val_splits = [train_splits.pop(k), train_splits.pop(k-1)]

    # Create an HCQT feature extraction module comprising
    # the first five harmonics and a sub-harmonic, where each
    # harmonic transform spans 4 octaves w/ 3 bins per semitone
    data_proc = HCQT(sample_rate=ppn.GSET_SAMPLE_RATE,
                     hop_length=ppn.GSET_HOP_LEN,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    print(f"Preparing the trian set in {ppn.GSET_CACHE_TRAIN}")
    # Create a dataset corresponding to the training partition
    gset_train = GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
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
    gset_val = GuitarSetPPN(base_dir=ppn.GSET_BASE_DIR,
                         splits=val_splits,
                         num_frames=ppn.NUM_FRAMES,
                         data_proc=data_proc,
                         profile=profile,
                         store_data=True,
                         save_loc=ppn.GSET_CACHE_VAL,
                         seed=ppn.RANDOM_SEED + 1)

    # Create a PyTorch data loader for the dataset
    val_loader = torch.utils.data.DataLoader(dataset=gset_val,
                              batch_size=ppn.BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)

    model = FretNetCrepe(
            dim_in=ppn.HCQT_DIM_IN,
            in_channels=ppn.HCQT_NO_HARMONICS,
            no_pitch_bins=ppn.PITCH_BINS
            )

    model.change_device(device=0)

    print("Starting the training")
    train(train_loader, val_loader, model, model_dir)

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

    # steps progress bar on the screen
    progress = tqdm(range(ppn.STEPS * 2))

    # train loss message on the screen
    tloss_log = tqdm(total=0, position=1, bar_format='{desc}')

    # evaluation loss message on the screen
    eloss_log = tqdm(total=0, position=2, bar_format='{desc}')


    while step < ppn.STEPS * 2:
        model.train()

        train_losses = []

        # Loop through the dataset
        for batch in train_loader:
            # Unpack batch
            features = batch[ppn.KEY_FEATURES]
            pitch_array = batch[ppn.KEY_PITCH_ARRAY]

            with torch.autocast(model.device.type):

                # Forward pass
                output = model(features.to(model.device))

                # Compute losses
                loss = ppn.train.loss(output[ppn.KEY_PITCH_LOGITS], pitch_array.to(model.device))
                train_losses.append(loss.item())

            # Zero the accumulated gradients
            optimizer.zero_grad()

            # Backward pass
            scaler.scale(loss).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            step += 1

            progress.update()

        train_losses = sum(train_losses) / len(train_losses)

        # log the trian loss
        writer.add_scalar(tag='train_loss_' + ppn.LOSS_BCE, 
                          scalar_value=train_losses, 
                          global_step=step)
        tloss_log.set_description(f"Train loss: {train_losses}")


        eval_loss = evaluate(val_loader, model)

        # log the evaluation loss
        writer.add_scalar(tag='eval_loss_' + ppn.LOSS_BCE,
                          scalar_value=eval_loss,
                          global_step=step)
        eloss_log.set_description(f"Evaluation loss: {eval_loss}")

        epoch += 1

    progress.close()

    # Save final model
    torchutil.checkpoint.save(
        log_dir / f'{step:08d}.pt',
        model,
        optimizer,
        step=step,
        epoch=epoch)


def evaluate(
        loader: torch.utils.data.DataLoader,
        model):
    """
    Perform model evaluation.
    """
    eval_losses = []

    with torch.no_grad():
        model.eval()

        for batch in loader:
            features = batch[ppn.KEY_FEATURES].to(device=model.device)
            pitch_array = batch[ppn.KEY_PITCH_ARRAY].to(device=model.device)

            # set the pitch names to something
            output = model(features)

            # Compute losses
            loss = ppn.train.loss(output[ppn.KEY_PITCH_LOGITS], pitch_array)

            eval_losses.append(loss.item())

    eval_losses = sum(eval_losses) / len(eval_losses)

    return eval_losses
