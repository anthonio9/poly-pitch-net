import poly_pitch_net as ppn
from poly_pitch_net.datasets.guitarset import GuitarSetPPN
from poly_pitch_net.models import FretNetCrepe
from poly_pitch_net.models import MonoPitchNet1D
from poly_pitch_net.models import MonoPitchNet2D
import amt_tools.tools
from amt_tools.features import HCQT

from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import random
import librosa
import torchutil
import wandb


def prepare_and_run(model_type: str,
        gpu: int = None, 
        loss: str = ppn.LOSS_ONE_HOT,
        register_silence: bool = False,
        use_wandb: bool = False):

    log_wandb = None
    
    if use_wandb:
        wandb.login()

        log_wandb = wandb.init(
            # Set the project where this run will be logged
            project="MonoPitchNet",

            # Track hyperparameters and run metadata
            config={
                "learning_rate": ppn.LEARNING_RATE,
                "epochs": ppn.STEPS // 2,
                "register_silence" : register_silence,
                "loss" : loss,
                "pitch_bins" : ppn.PITCH_BINS,
                "model_type" : model_type,
            })

    if not register_silence:
        print("silence wont be registered")

    run(model_type,
        gpu,
        loss,
        register_silence,
        log_wandb)


def run(model_type: str,
        gpu: int = None, 
        loss: str = ppn.LOSS_ONE_HOT,
        register_silence: bool = False,
        log_wandb=None):

    if 'mono1d' in model_type:
        EX_NAME = '_'.join([MonoPitchNet1D.model_name(),
                            GuitarSetPPN.dataset_name(),
                            HCQT.features_name()])

        model = MonoPitchNet1D(
                dim_in=ppn.HCQT_DIM_IN,
                no_pitch_bins=ppn.PITCH_BINS,
                register_silence=register_silence)

    elif 'mono2d' in model_type:
        EX_NAME = '_'.join([MonoPitchNet2D.model_name(),
                            GuitarSetPPN.dataset_name(),
                            HCQT.features_name()])

        model = MonoPitchNet2D(
                no_pitch_bins=ppn.PITCH_BINS,
                register_silence=register_silence,
                string=3,
                cqt=True,
                audio=True)


    elif 'poly' in model_type:
        EX_NAME = '_'.join([FretNetCrepe.model_name(),
                            GuitarSetPPN.dataset_name(),
                            HCQT.features_name()])

        model = FretNetCrepe(
                dim_in=ppn.HCQT_DIM_IN,
                in_channels=ppn.HCQT_NO_HARMONICS,
                no_pitch_bins=ppn.PITCH_BINS)

    else:
        print(f"{model_type} is not supported!")
        return

    if log_wandb is not None:
        log_wandb.config["architecture"] = str(model)

    # Create the root directory for the experiment files
    experiment_dir = ppn.tools.misc.get_project_root().parent / '..' / 'generated' / 'experiments' / EX_NAME

    # Create a log directory for the training experiment
    model_dir = experiment_dir / 'models'

    train_loader = ppn.datasets.loader('train')
    val_loader = ppn.datasets.loader('val')


    model.change_device(device=gpu)

    print("Starting the training")
    train(train_loader, val_loader, model, model_dir, loss_type=loss, log_wandb=log_wandb)

def train(
        train_loader,
        val_loader,
        model,
        log_dir,
        loss_type: str = ppn.LOSS_ONE_HOT,
        log_wandb=None):

    # Initialize a writer to log any reported results
    writer = SummaryWriter(log_dir)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ppn.LEARNING_RATE)

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    step, epoch = 0, 0

    # steps progress bar on the screen
    progress = tqdm(range(ppn.STEPS // 2))

    # train loss message on the screen
    tloss_log = tqdm(total=0, position=1, bar_format='{desc}')

    # evaluation loss message on the screen
    eloss_log = tqdm(total=0, position=2, bar_format='{desc}')


    while step < ppn.STEPS // 2:
        model.train()

        train_losses = []

        # Loop through the dataset
        for batch in train_loader:

            with torch.autocast(model.device.type):

                # Forward pass
                output = model(batch)

                # pitch array is already pre-processed
                pitch_array = batch[ppn.KEY_PITCH_ARRAY]
                pitch_array = pitch_array.to(model.device)

                # Compute losses
                loss = ppn.train.loss(
                        model, 
                        output[ppn.KEY_PITCH_LOGITS],
                        pitch_array.to(model.device),
                        loss_type=loss_type)
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

        # log the train loss
        writer.add_scalar(tag='train_loss_' + ppn.LOSS_BCE, 
                          scalar_value=train_losses, 
                          global_step=step)
        tloss_log.set_description(f"Train loss: {train_losses}")


        # because there's 8 batches in every train loader
        if step % 80 == 0:
            eval_loss, metric_dict = evaluate(val_loader, model, loss_type, log_wandb)
        else:
            eval_loss, metric_dict = evaluate(val_loader, model, loss_type, None)

        # log the evaluation loss
        writer.add_scalar(tag='eval_loss_' + ppn.LOSS_BCE,
                          scalar_value=eval_loss,
                          global_step=step)
        write_metrics(writer, step, metric_dict)

        if log_wandb is not None:
            metric_dict["train_loss"] = train_losses
            metric_dict["eval_loss"] = eval_loss
            log_wandb.log(metric_dict)

        eloss_log.set_description(
                f'Evaluation loss: {eval_loss} '
                f'acc: {metric_dict["accuracy"]} '
                f'rmse: {metric_dict["RMSE"]} '
                f'rpa: {metric_dict["RPA"]}')

        epoch += 1

    progress.close()

    # Save final model
    torchutil.checkpoint.save(
        log_dir / f'model_{model.model_name().lower()}_{step:08d}.pt',
        model,
        optimizer,
        step=step,
        epoch=epoch)


def write_metrics(writer: SummaryWriter, step: int, metrics: dict):
    # log the evaluation loss
    for key, val in metrics.items():
        writer.add_scalar(tag='eval_' + key,
                          scalar_value=val,
                          global_step=step)


def evaluate(
        loader: torch.utils.data.DataLoader,
        model,
        loss_type: str = ppn.LOSS_ONE_HOT, 
        log_wandb=None):
    """
    Perform model evaluation.
    """
    eval_losses = []
    metrics = ppn.evaluate.metrics.Metrics(20)

    with torch.no_grad():
        model.eval()

        output = None
        pitch_array = None
        batch = None

        for batch in loader:
            # set the pitch names to something
            output = model(batch)

            # process into pitch cents
            output = model.post_proc(output)

            # pitch array is already pre-processed
            pitch_array = batch[ppn.KEY_PITCH_ARRAY]
            pitch_array = pitch_array.to(model.device)

            # remove frames without pitch / with silence
            cents = output[ppn.KEY_PITCH_ARRAY_CENTS]
            cents = cents[pitch_array != 0]
            pitch_array_no_silence = pitch_array[pitch_array != 0]
            
            # print(f"cents zeros: {cents.numel() - cents.count_nonzero()}")

            # get metrics
            metrics.update(cents,
                           ppn.tools.frequency_to_cents(
                               pitch_array_no_silence, 
                               register_silence=model.register_silence))

            # Compute losses
            loss = ppn.train.loss(model, output[ppn.KEY_PITCH_LOGITS], pitch_array,
                    loss_type=loss_type)

            eval_losses.append(loss.item())

        if log_wandb is not None:
            fig = ppn.evaluate.plot_logits(
                    output[ppn.KEY_PITCH_LOGITS],
                    batch[ppn.KEY_PITCH_ARRAY],
                    loss_type = loss_type)

            fig2 = ppn.evaluate.plotly_pitch(
                    output[ppn.KEY_PITCH_ARRAY_HZ],
                    batch[ppn.KEY_PITCH_ARRAY],
                    batch[ppn.KEY_TIMES])

            log_wandb.log({"logits_chart" : fig})
            log_wandb.log({"pitch_chart" : fig2})

    eval_losses = sum(eval_losses) / len(eval_losses)

    return eval_losses, metrics.get_metrics()
