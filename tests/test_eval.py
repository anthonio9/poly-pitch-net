import poly_pitch_net as ppn


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
                                 times=times)
