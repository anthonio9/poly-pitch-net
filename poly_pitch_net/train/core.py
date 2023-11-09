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

    train_loader = ppn.datasets.loader('train')
    val_loader = ppn.datasets.loader('val')

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
