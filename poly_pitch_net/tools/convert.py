import poly_pitch_net as ppn
import guitar_transcription_continuous.utils as utils
from guitar_transcription_continuous.estimators import StackedPitchListTablatureWrapper
from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedOffsetsWrapper, \
                                 StackedNoteTranscriber
import amt_tools.tools as tools

import numpy as np


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    # return penn.CENTS_PER_BIN * bins
    return ppn.CENTS_PER_BIN * bins


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return ppn.OCTAVE * torch.log2(frequency / ppn.FMIN)


def tablature_rel_to_pitch(batch, profile):

    breakpoint()
    tab = batch[tools.KEY_TABLATURE]
    tab_rel = batch[utils.KEY_TABLATURE_REL]
    midi = batch[tools.KEY_MULTIPITCH]

    # tab_shape = tab_rel.shape
    # new_shape = (
    #         tab_shape[0],
    #         tab_shape[1],
    #         profile.get_num_dofs() # should be 6
    #         profile.get_num_frets() + 1 # should be 20
    #         )
    # tab_rel = tab_rel.reshape(shape=new_shape)

    pitch = tools.logistic_to_stacked_multi_pitch(tab_rel, profile, silence=False)

    # TO BO CONTINUED

def extract_stacked_pitch_list_jams(jam, times=None, uniform=True):
    """
    Extract pitch lists spread across slices (e.g. guitar strings) from JAMS annotations into a dictionary.

    Parameters
    ----------
    jam : JAMS object
      JAMS file data
    times : ndarray or None (optional) (N)
      Time in seconds for resampling
      N - number of time samples
    uniform : bool
      Whether to place annotations on a uniform time grid

    Returns
    ----------
    stacked_pitch_list : dict
      Dictionary containing (slice -> (times, pitch_list)) pairs
    """

    # Extract all of the pitch annotations
    pitch_data_slices = jam.annotations[tools.constants.JAMS_PITCH_HZ]

    # Obtain the number of annotations
    stack_size = len(pitch_data_slices)

    # Initialize a dictionary to hold the pitch lists
    stacked_pitch_list = dict()

    # Loop through the slices of the stack
    for slc in range(stack_size):
        # Extract the pitch list pertaining to this slice
        slice_pitches = pitch_data_slices[slc]

        # Extract the string label for this slice
        string = slice_pitches.annotation_metadata[tools.constants.JAMS_STRING_IDX]

        # Initialize an array/list to hold the times/frequencies associated with each observation
        entry_times, slice_pitch_list = np.zeros(1), list()

        # Loop through the pitch observations pertaining to this slice
        for pitch in slice_pitches:
            # Extract the pitch
            freq = np.array([pitch.value['frequency']])

            # Don't keep track of zero or unvoiced frequencies
            if np.sum(freq) == 0 or not pitch.value['voiced']:
                freq = np.zeros(1)

            # Append the observation time
            entry_times = np.append(entry_times, pitch.time)
            # Append the frequency
            slice_pitch_list.append(freq)

        # Sort the pitch list before resampling just in case it is not already sorted
        entry_times, slice_pitch_list = tools.utils.sort_pitch_list(entry_times, slice_pitch_list)

        if uniform:
            # Align the pitch list with a uniform time grid
            entry_times, slice_pitch_list = tools.utils.time_series_to_uniform(times=entry_times,
                                                                         values=slice_pitch_list,
                                                                         duration=jam.file_metadata.duration)

        if times is not None:
            # Resample the observation times if new times are specified
            slice_pitch_list = resample_multipitch(entry_times, slice_pitch_list, times)
            # Overwrite the entry times with the specified times
            entry_times = times

        # Add the pitch list to the stacked pitch list dictionary under the slice key
        stacked_pitch_list.update(tools.utils.pitch_list_to_stacked_pitch_list(entry_times, slice_pitch_list, string))

    return stacked_pitch_list
