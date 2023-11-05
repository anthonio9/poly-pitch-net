import amt_tools.tools
from amt_tools.features import HCQT
from poly_pitch_net.datasets.guitarset import GuitarSetPPN
import librosa
import poly_pitch_net as ppn
import torch


def main():
    # run evaluation on a single value from a batch 

    profile = amt_tools.tools.GuitarProfile(num_frets=19)

    # Create an HCQT feature extraction module comprising
    # the first five harmonics and a sub-harmonic, where each
    # harmonic transform spans 4 octaves w/ 3 bins per semitone
    data_proc = HCQT(sample_rate=ppn.GSET_SAMPLE_RATE,
                     hop_length=ppn.GSET_HOP_LEN,
                     fmin=librosa.note_to_hz('E2'),
                     harmonics=[0.5, 1, 2, 3, 4, 5],
                     n_bins=144, bins_per_octave=36)

    # create a train_loader
    gset_train = GuitarSetPPN(
            base_dir=ppn.GSET_BASE_DIR,
            splits=[GuitarSetPPN.available_splits().pop(0)],
            num_frames=ppn.NUM_FRAMES,
            profile=profile,
            data_proc=data_proc,
            reset_data=False, # set to true in the future trainings
            save_data=True, # set to true in the future trainings
            save_loc=ppn.GSET_CACHE_PYTEST,
            seed=ppn.RANDOM_SEED)

    # Create a PyTorch data loader for the dataset
    train_loader = torch.utils.data.DataLoader(dataset=gset_train,
                              batch_size=1,
                              shuffle=True,
                              drop_last=True)

    train_loader = iter(train_loader)
    batch = next(train_loader)

    features = batch[ppn.KEY_FEATURES].numpy()[0, 0, :, :]
    pitch = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    times = batch[ppn.KEY_TIMES].numpy()[0, :]

    ppn.evaluate.plot_poly_pitch(freq=features,
                                 pitch_hat=pitch,
                                 times=times)
    

main()
