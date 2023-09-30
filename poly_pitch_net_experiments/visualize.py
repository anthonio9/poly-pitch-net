import amt_tools.tools as tools
import matplotlib.pyplot as plt
import librosa as lbs
import numpy as np
from scipy import signal


def plot_stacked_pitch_list_with_spectrogram(audio, features, sample_rate, hop_length, stacked_pitch_list, hertz=False,
                                             point_size=5, include_axes=True,
                                             x_bounds=None, y_bounds=None,
                                             colors=None, labels=None):

    fig, ax = plt.subplots()
    fig.tight_layout()

    # STFT
    f, t, Zxx = signal.stft(audio, sample_rate, nperseg=hop_length*9, noverlap=hop_length*8)
    ax.pcolormesh(t, f, np.abs(Zxx))

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
    for idx, slc in enumerate(stacked_pitch_list.keys()):
        # Get the times and pitches from the slice
        times, pitch_list = stacked_pitch_list[slc]
        # Determine the color to use when plotting the slice
        color = 'k' if colors is None else colors[idx]
        # Determine the label to use when plotting the slice
        label = None if labels is None else labels[idx]

        #times, pitches = tools.utils.unroll_pitch_list(times, pitch_list)

        #try:
        #    print(f"STRING {slc} - min: {pitches.min()}, max: {pitches.max()}")
        #except ValueError:
        #    pass
    
        #if len(collections) and idx < len(collections):
        ## Re-use the selected scatter collection and plot the new points
        #    collections[idx].set_offsets(np.c_[times, pitches])
        #else:
        #    # Plot the points as a new collection
        #    ax.scatter(times, pitches, s=point_size, color=color, marker='o', label=label, alpha=1.0)
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
