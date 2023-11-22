import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import librosa
import plotly.express as px
import torch


def onehot_with_ignore_label(labels, num_class, ignore_label):
    dummy_label = num_class + 1

    # set the mask for the ingored labels
    mask = labels == ignore_label
    modified_labels = labels.clone()

    # set ignored labels to max value
    modified_labels[mask] = num_class

    # one-hot encode the modified labels
    one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=dummy_label)

    # remove the last row in the one-hot encoding
    one_hot_labels = one_hot_labels[:, :, :-1]
    return one_hot_labels


def plot_logits(logits: torch.Tensor, pitch_array: torch.Tensor, string: int=-1):
    assert len(logits.shape) == 3
    assert len(pitch_array.shape) == 3

    # convert to numpy
    logits = logits.cpu().numpy()
    pitch_array = pitch_array.cpu().numpy()

    if string >= 0:
        pitch_array = pitch_array[:, string, :]

    # generate the pitch_array bins
    pitch_bins = ppn.tools.frequency_to_bins(pitch_array)
    pitch_bins[pitch_array == 0] = -1
    pitch_bins_1hot = onehot_with_ignore_label(
            pitch_bins, ppn.PITCH_BINS, -1)

    logits = logits.permute(0, 2, 1)
    logits = logits[0, :, :]
    pitch_bins_1hot = pitch_bins_1hot[0, :, :]

    assert logits.shape == pitch_bins_1hot.shape

    fix = px.imxshow(
            logits, 
            color_continuous_scale=px.colors.sequential.Cividis_r)

    fix.show()



def pitch_to_lines(pitch_array, times, linestyle='solid', label='String'):
    """Create pitch lines

    Args
        pitch_array (numpy array)
            Predicted pitch array of shape [C, T]. One string arrays are supported,
            just make sure that the shape is correct, for one string [1, T]
        times (numpy array) 
            Array with timestamps, shape [T]
    """
    assert len(pitch_array.shape) == 2
    assert len(times.shape) == 1

    # replace 0s with -1 (for masking later)
    pitch_array[pitch_array == 0] = -1

    # create LineCollection segments 
    # https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
    segs = [np.column_stack([times, pitch]) for pitch in list(pitch_array)]

    # mask spots where pitch wasn't recognized
    segs = np.ma.masked_equal(segs, -1)

    # *colors* is sequence of rgba tuples.
    # colors copied from rcParams['axes.prop_cycle']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = colors[:pitch_array.shape[0]]

    # *linestyle* is a string or dash tuple. Legal string values are
    # solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq) where
    # onoffseq is an even length tuple of on and off ink in points.  If linestyle
    # is omitted, 'solid' is used.
    # See `matplotlib.collections.LineCollection` for more information.

    lines = LineCollection(segs, linewidths=(1.5),
                               colors=colors, linestyle=linestyle)

    def make_proxy(color, **kwargs):
        return Line2D([0, 1], [0, 1], color=color, **kwargs)

    proxies = [make_proxy(color, linestyle=linestyle, linewidth=2) for color in colors]
    string_labels = [f"{label} {st}" for st in range(pitch_array.shape[0])]

    return lines, proxies, string_labels



def plot_poly_pitch(
        freq, 
        pitch_hat, 
        times, 
        pitch_gt=None,
        freq_type='STFT',
        show_plot=True):
    """
    Plot the pitch on the frequency graph with matplotlib.

    Args:
        freq (numpy array)
            Spectrogram array, of shape [F, T]
        pitch_hat (numpy array)
            Predicted pitch array of shape [C, T]. One string arrays are supported,
            just make sure that the shape is correct, for one string [1, T]
        times (numpy array) 
            Array with timestamps, shape [T]
        pitch_gt (numpy array)
            Optional array with target pitch, shape [C, T]

        F - number of the frequency bins
        C - number of the guitar strings
        T - number of the time stamps
    """
    if freq is not None:
        assert len(freq.shape) == 2

    assert len(pitch_hat.shape) == 2
    assert len(times.shape) == 1

    fig, ax = plt.subplots()

    # plot spectrogram
    hop_length = ppn.GSET_HOP_LEN 
    dB = freq

    if freq is not None and freq_type == 'STFT':
        dB = librosa.amplitude_to_db(np.abs(dB), ref=np.max)
        img = librosa.display.specshow(dB, y_axis='linear', x_axis='time',
                                       sr=ppn.GSET_SAMPLE_RATE, ax=ax, x_coords=times)
    if freq is not None and freq_type == 'HCQT':
        img = librosa.display.specshow(dB, y_axis='cqt_hz', x_axis='time',
                                       ax=ax, x_coords=times, bins_per_octave=36,
                                       fmin=freq_vec[0], fmax=freq_vec[-1])

    # get the min and max of the pitch values
    pitch_hat_no_zeros = pitch_hat.flatten()
    pitch_hat_no_zeros = pitch_hat_no_zeros[pitch_hat_no_zeros != 0]

    try:
        mins = [pitch_hat_no_zeros.min()]
        maxs = [pitch_hat_no_zeros.max()]
    except ValueError:
        mins = [0]
        maxs = [ppn.GSET_SAMPLE_RATE // 2]

    print(f"min pitch {min(mins)} Hz")
    print(f"max pitch {max(maxs)} Hz")

    lines, proxies, string_labels = pitch_to_lines(pitch_hat, times, linestyle='dashed')
    ax.add_collection(lines)
    ax.set_title('Pitch')

    if pitch_gt is not None:
        lines_gt, proxies_gt, string_labels_gt = pitch_to_lines(pitch_gt, 
                                                                times, 
                                                                linestyle='dashdot',
                                                                label='GT String')
        ax.add_collection(lines_gt)

        proxies.extend(proxies_gt)
        string_labels.extend(string_labels_gt)
        
        # correct the limits with ground truth taken into account
        pitch_gt_no_zeros = pitch_gt.flatten()
        pitch_gt_no_zeros = pitch_gt_no_zeros[pitch_gt_no_zeros != 0]
        mins.append(pitch_gt_no_zeros.min())
        maxs.append(pitch_gt_no_zeros.max())

        print(f"min gt pitch {pitch_gt_no_zeros.min()} Hz")
        print(f"max gt pitch {pitch_gt_no_zeros.max()} Hz")

    offset = max(maxs) - min(mins) 
    offset *= 0.7

    # We need to set the plot limits, they will not autoscale
    ax.set_xlim(times.min(), times.max())
    # ax.set_ylim(ymin=min(mins) - offset, ymax=max(maxs) + offset)
    ax.set_ylim(ymax=max(maxs) + offset)

    # Manually adding artists doesn't rescale the plot, so we need to autoscale
    # ax.autoscale()

    ax.legend(proxies, string_labels, bbox_to_anchor=(1.01, 1),
                         loc='upper left', borderaxespad=0.)
    # fig.colorbar(img, ax=ax, format="%+2.f dB", orientation="horizontal")

    if show_plot:
        plt.show()

    return fig


def plot_mono_pitch(
    freq, 
    pitch_hat, 
    times, 
    pitch_gt=None,
    freq_type='STFT',
    show_plot=True):
    """Plot mono pitch with ground truth

    Args:
        freq (numpy array)
            Spectrogram or different kind of transform
            Should be of shape [F, T]
        pitch_hat (numpy array)
            Array with predicted pitch values in Hz from one string
            Should be of shape [T]
        times (numpy array)
            Array with time interval labels (in sec)
            Should be of shape [T]
        pitch_gt (numpy array)
            Array with ground truth pitch values in Hz from one string
            Should be of shape [T]

    Returns:
        fig (matplotlib.plt figure)
            Matplotlib figure containing the pitch plot
    """
    assert len(pitch_hat.shape) == 1

    pitch_hat = np.expand_dims(pitch_hat, axis=0)

    if pitch_gt is not None:
        if len(pitch_gt.shape) == 1:
            pitch_gt = np.expand_dims(pitch_gt, axis=0)

        assert len(pitch_gt.shape) == 2

    return plot_poly_pitch(freq, pitch_hat, times, pitch_gt, freq_type, show_plot=show_plot)
