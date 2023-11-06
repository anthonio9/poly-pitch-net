import amt_tools.tools
import poly_pitch_net as ppn

from pathlib import Path
import torch
import torchutil


def run_evaluation(
        model_path: Path,
        gpu: int = None):

    # load the model if path exist
    if not Path.exists(model_path):
        print(f"Given model path '{model_path}' does not exist.")
        return 1

    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    model = FretNetCrepe(
            dim_in=ppn.HCQT_DIM_IN,
            in_channels=ppn.HCQT_NO_HARMONICS,
            no_pitch_bins=ppn.PITCH_BINS
            )

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ppn.LEARNING_RATE)

    model, optimizer, state = torchutil.checkpoint.load(
            model_path,
            model,
            optimizer)

    loader = ppn.datasets.loader('val')

    loader = iter(loader)
    batch = next(loader)

    with torch.no_grad():
        model.to(device)
        features = batch[ppn.KEY_FEATURES].to(device)
        output = model(features)
        output = model.post_proc(output)

    features = batch[ppn.KEY_FEATURES].numpy()[0, 0, :, :]
    pitch_gt = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    times = batch[ppn.KEY_TIMES].numpy()[0, :]

    pitch = output[ppn.KEY_PITCH_WG_AVG].numpy()[0, :, :]

    ppn.evaluate.plot_poly_pitch(freq=features,
                                 pitch_hat=pitch,
                                 pitch_gt=pitch_gt,
                                 times=times)
    

