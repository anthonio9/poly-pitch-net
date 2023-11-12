import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

import poly_pitch_net as ppn


def pitch_to_lines(pitch_array, times, linestyle='solid', label='String'):
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
        pitch_gt=None):
    """
    Plot the pitch on the frequency graph with matplotlib.
    """
    # get the min and max of the pitch values
    pitch_hat_no_zeros = pitch_hat.flatten()
    pitch_hat_no_zeros = pitch_hat_no_zeros[pitch_hat_no_zeros != 0]
    mins = [pitch_hat_no_zeros.min()]
    maxs = [pitch_hat_no_zeros.max()]

    fig, ax = plt.subplots()

    lines, proxies, string_labels = pitch_to_lines(pitch_hat, times, linestyle='dashed')
    ax.add_collection(lines)
    ax.set_title('Pitch')

    if pitch_gt is not None:
        lines_gt, proxies_gt, string_labels_gt = pitch_to_lines(pitch_gt, 
                                                                times, 
                                                                linestyle='dotted',
                                                                label='GT String')
        ax.add_collection(lines_gt)

        proxies.extend(proxies_gt)
        string_labels.extend(string_labels_gt)
        
        # correct the limits with ground truth taken into account
        pitch_gt_no_zeros = pitch_gt.flatten()
        pitch_gt_no_zeros = pitch_gt_no_zeros[pitch_gt_no_zeros != 0]
        mins.append(pitch_gt_no_zeros.min())
        maxs.append(pitch_gt_no_zeros.max())

    offset = max(maxs) - min(mins) 
    offset *= 0.2

    # We need to set the plot limits, they will not autoscale
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(pitch_hat_no_zeros.min() - offset, pitch_hat_no_zeros.max() + offset)

    # Manually adding artists doesn't rescale the plot, so we need to autoscale
    ax.autoscale()

    ax.legend(proxies, string_labels, bbox_to_anchor=(1.01, 1),
                         loc='upper left', borderaxespad=0.)
    plt.show()


def plot_mono_pitch(
    freq, 
    pitch_hat, 
    times, 
    pitch_gt=None):
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
    """
    assert len(pitch_hat.shape) == 1
    assert len(pitch_gt.shape) == 1

    pitch_hat = np.expand_dims(pitch_hat, axis=0)
    pitch_gt = np.expand_dims(pitch_gt, axis=0)

    plot_poly_pitch(freq, pitch_hat, times, pitch_gt)
