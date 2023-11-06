import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

import poly_pitch_net as ppn


def plot_poly_pitch(
        freq, 
        pitch_hat, 
        times, 
        pitch_gt=None):
    """
    Plot the pitch on the frequency graph with matplotlib.
    """
    pitch_hat_no_zeros = pitch_hat.flatten()
    pitch_hat_no_zeros = pitch_hat_no_zeros[pitch_hat_no_zeros != 0]
    offset = pitch_hat_no_zeros.max() - pitch_hat_no_zeros.min() 
    offset *= 0.2

    # replace 0s with -1 (for masking later)
    pitch_hat[pitch_hat == 0] = -1

    # create LineCollection segments 
    # https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
    segs = [np.column_stack([times, pitch]) for pitch in list(pitch_hat)]

    # mask spots where pitch wasn't recognized
    segs = np.ma.masked_equal(segs, -1)

    # We need to set the plot limits, they will not autoscale
    fig, ax = plt.subplots()
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(pitch_hat_no_zeros.min() - offset, pitch_hat_no_zeros.max() + offset)

    # *colors* is sequence of rgba tuples.
    # *linestyle* is a string or dash tuple. Legal string values are
    # solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq) where
    # onoffseq is an even length tuple of on and off ink in points.  If linestyle
    # is omitted, 'solid' is used.
    # See `matplotlib.collections.LineCollection` for more information.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    lines = LineCollection(segs, linewidths=(1.5),
                               colors=colors, linestyle='solid')
    ax.add_collection(lines)
    ax.set_title('Pitch')

    # Manually adding artists doesn't rescale the plot, so we need to autoscale
    ax.autoscale()

    def make_proxy(color, **kwargs):
        return Line2D([0, 1], [0, 1], color=color, **kwargs)

    proxies = [make_proxy(color, linewidth=5) for color in colors]
    string_labels = [f"String {st}" for st in range(pitch_hat.shape[0])]
    ax.legend(proxies, string_labels, bbox_to_anchor=(1.01, 1),
                         loc='upper left', borderaxespad=0.)
    # fig.legend(proxies, string_labels, loc='outside right upper')

    plt.show()
