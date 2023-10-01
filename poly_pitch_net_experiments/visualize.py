import amt_tools.tools as tools
import matplotlib.pyplot as plt
import librosa as lbs
import numpy as np
from scipy import signal


def plot_stacked_pitch_list_with_spectrogram(audio, ground_truth, sample_rate, hop_length, stacked_pitch_list, hertz=False,
                                             point_size=5, include_axes=True,
                                             x_bounds=None, y_bounds=None,
                                             colors=None, labels=None):

    fig, ax = plt.subplots()
    fig.tight_layout()

    # STFT
    f, t, Zxx = signal.stft(audio, sample_rate, nperseg=hop_length*9, noverlap=hop_length*8)
    ax.pcolormesh(t, f, np.log(np.abs(Zxx)))

    # cqt = features[tools.KEY_FEATS][0, :, :]
    # time = features[tools.KEY_TIMES]

    # IMSHOW
    #freq = lbs.cqt_frequencies(n_bins=144, fmin=lbs.note_to_hz('E2'),
    #        bins_per_octave=36)
    #ax.imshow(np.flip(cqt, axis=0), zorder=0, extent=[time[0], time[-1], freq[0]-600, freq[-1]-600], aspect='auto')
    #ax.set_xlabel('time')
    #ax.set(xticks=np.arange(0, len(time))[::50], xticklabels=time.astype(float)[::50])
    #ax.set(yticks=np.arange(0, 144), yticklabels=freq)

    # PCOLORMESH
    #time = np.append(time, [time[-1] + 512/sample_rate])
    #freq = lbs.cqt_frequencies(n_bins=145, fmin=lbs.note_to_hz('E2'),
    #        bins_per_octave=36)
    #ax.pcolormesh(time, freq, cqt)
    collections = ax.collections

    # Loop through the stack of pitch lists, keeping track of the index
    #for idx, slc in enumerate(stacked_pitch_list.keys()):
    #    # Get the times and pitches from the slice
    #    times, pitch_list = stacked_pitch_list[slc]
    #    # Determine the color to use when plotting the slice
    #    color = 'k' if colors is None else colors[idx]
    #    # Determine the label to use when plotting the slice
    #    label = None if labels is None else labels[idx]

    #    ## Use the pitch_list plotting function
    #    fig = tools.plot_pitch_list(times=times,
    #                                pitch_list=pitch_list,
    #                                hertz=hertz,
    #                                point_size=point_size,
    #                                include_axes=include_axes,
    #                                x_bounds=x_bounds,
    #                                y_bounds=y_bounds,
    #                                color=color,
    #                                label=label,
    #                                idx=idx,
    #                                fig=fig)

    stacked_pitch_list_gt = tools.unpack_stacked_representation(
            ground_truth[tools.KEY_PITCHLIST])

    keys_to_pop = []
    for idx, slc in enumerate(stacked_pitch_list_gt.keys()):
        slc_len = [len(value) for value in stacked_pitch_list_gt[slc]]
        slc_len = sum(slc_len)
        if slc_len == 0:
            keys_to_pop.append(slc)

    for key in keys_to_pop:
        stacked_pitch_list_gt.pop(key)

    stacked_pitch_list_gt = tools.stacked_pitch_list_to_hz(
        stacked_pitch_list_gt)

    # Loop through the stack of pitch lists, keeping track of the index
    for idx, slc in enumerate(stacked_pitch_list_gt.keys()):
        # Get the times and pitches from the slice
        times, pitch_list = stacked_pitch_list_gt[slc]
        # Determine the color to use when plotting the slice
        color = 'b' if colors is None else colors[idx]
        # Determine the label to use when plotting the slice
        label = None if labels is None else labels[idx]

        ## Use the pitch_list plotting function
        fig = tools.plot_pitch_list(times=times,
                                    pitch_list=pitch_list,
                                    hertz=hertz,
                                    point_size=point_size,
                                    include_axes=include_axes,
                                    x_bounds=x_bounds,
                                    y_bounds=y_bounds,
                                    color=color,
                                    label=label,
                                    idx=idx,
                                    fig=fig)

    plt.show()

    return fig
