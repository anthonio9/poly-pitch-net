import amt_tools.tools
import poly_pitch_net as ppn
from poly_pitch_net.models import FretNetCrepe
from poly_pitch_net.models import MonoPitchNet1D

from pathlib import Path
import torch
import torchutil


def run_evaluation(
        model_path: Path,
        gpu: int = None,
        register_silence: bool = False):

    # load the model if path exist
    if not Path.exists(model_path):
        print(f"Given model path '{model_path}' does not exist.")
        return 1

    if 'monopitchnet1d' in model_path.stem:
        model = MonoPitchNet1D(
                dim_in=ppn.HCQT_DIM_IN,
                no_pitch_bins=ppn.PITCH_BINS,
                register_silence=register_silence
                )
    else:
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

    iloader = iter(loader)
    batch = next(iloader)

    with torch.no_grad():
        model.change_device(gpu)
        features = batch[ppn.KEY_FEATURES].to(model.device)

        if 'MonoPitchNet1D' in model.model_name():
            # choose HCQT channel 0
            features = features[:, 0, :, :]

        output = model(features)
        output[ppn.KEY_PITCH_LOGITS] = torch.nn.functional.sigmoid(output[ppn.KEY_PITCH_LOGITS])
        output = model.post_proc(output)
        loss = ppn.train.evaluate(loader, model)

        print(f"evaluation BCE loss: {loss}")

    features = batch[ppn.KEY_FEATURES].cpu().numpy()[0, 0, :, :]
    pitch_gt = batch[ppn.KEY_PITCH_ARRAY].cpu().numpy()[0, :, :]
    times = batch[ppn.KEY_TIMES].cpu().numpy()[0, :]

    if 'MonoPitchNet1D' in model.model_name():
        # choose string 3
        pitch = output[ppn.KEY_PITCH_ARRAY_CENTS].cpu().numpy()[3, :]
        pitch = ppn.tools.convert.cents_to_frequency(pitch)
        pitch_gt = pitch_gt[3, :]

        ppn.evaluate.plot_mono_pitch(freq=features,
                                     pitch_hat=pitch,
                                     pitch_gt=pitch_gt,
                                     times=times)
    else:
        pitch = output[ppn.KEY_PITCH_WG_AVG].cpu().numpy()[0, :, :]
        pitch = ppn.tools.convert.cents_to_frequency(pitch)

        ppn.evaluate.plot_poly_pitch(freq=features,
                                     pitch_hat=pitch,
                                     pitch_gt=pitch_gt,
                                     times=times)
    

