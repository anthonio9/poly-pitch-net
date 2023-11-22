import poly_pitch_net as ppn
import librosa
import torch


def test_plot_poly_pitch():
    loader = ppn.datasets.loader('pytest', data_proc_type=None)

    loader = iter(loader)
    batch = next(loader)

    features = batch[ppn.KEY_FEATURES].numpy()[0, 0, :, :]
    pitch_gt = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    times = batch[ppn.KEY_TIMES].numpy()[0, :]

    pitch = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    pitch[pitch != 0] += 10

    ppn.evaluate.plot_poly_pitch(features, pitch, times, pitch_gt)


def test_plot_mono_pitch():
    loader = ppn.datasets.loader('pytest', data_proc_type=None)

    loader = iter(loader)
    batch = next(loader)

    # choose string number 3
    features = batch[ppn.KEY_FEATURES].numpy()[0, 0, :, :]
    pitch_gt = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, 3, :]
    times = batch[ppn.KEY_TIMES].numpy()[0, :]

    # pitch = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    # pitch[pitch != 0] += 10
    ppn.evaluate.plot_mono_pitch(freq=features,
                                 pitch_hat=pitch_gt,
                                 times=times, 
                                 freq_type='STFT')

def test_plot_poly_pitch_stft():
    loader = ppn.datasets.loader('pytest', data_proc_type=None)

    loader = iter(loader)
    batch = next(loader)

    # choose string number 3
    audio = batch[ppn.KEY_AUDIO].numpy()[0, :]
    features = librosa.stft(audio, hop_length=ppn.GSET_HOP_LEN,
                            win_length=ppn.GSET_HOP_LEN * 4,
                            n_fft=ppn.GSET_HOP_LEN * 4)
    pitch_gt = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    times = batch[ppn.KEY_TIMES].numpy()[0, :]

    ppn.evaluate.plot_poly_pitch(freq=features,
                                 pitch_hat=pitch_gt,
                                 times=times, 
                                 freq_type='STFT')

def test_plot_mono_pitch_hcqt():
    loader = ppn.datasets.loader('pytest', data_proc_type='HCQT')

    loader = iter(loader)
    batch = next(loader)

    # choose string number 3
    features = batch[ppn.KEY_FEATURES].numpy()[0, 0, :, :]
    pitch = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, 3, :]
    pitch_gt = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    times = batch[ppn.KEY_TIMES].numpy()[0, :]

    ppn.evaluate.plot_mono_pitch(freq=features,
                                 pitch_hat=pitch,
                                 pitch_gt=pitch_gt,
                                 times=times,
                                 freq_type='HCQT')

def test_plot_poly_pitch_no_features():
    loader = ppn.datasets.loader('pytest', data_proc_type=None)

    loader = iter(loader)
    batch = next(loader)

    # choose string number 3
    pitch_gt = batch[ppn.KEY_PITCH_ARRAY].numpy()[0, :, :]
    times = batch[ppn.KEY_TIMES].numpy()[0, :]

    ppn.evaluate.plot_poly_pitch(freq=None,
                                 pitch_hat=pitch_gt,
                                 times=times, 
                                 freq_type='STFT')


def test_plot_logits():
    loader = ppn.datasets.loader('pytest', data_proc_type=None)

    loader = iter(loader)
    batch = next(loader)

    logits = torch.arange(ppn.PITCH_BINS) / ppn.PITCH_BINS
    logits = logits.expand(30, ppn.PITCH_BINS, 200)

    ppn.evaluate.plot_logits(logits, batch[ppn.KEY_PITCH_ARRAY], 3)
