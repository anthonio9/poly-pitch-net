from guitar_transcription_continuous.datasets import GuitarSetPlus
from amt_tools.datasets import GuitarSet
import amt_tools.tools as tools
from amt_tools.datasets import TranscriptionDataset
import jams
import numpy as np
import os
from copy import deepcopy

import poly_pitch_net as ppn


GSET_SAMPLE_RATE = 44100
GSET_HOP_LEN = 256
GSET_TIME_STEP = GSET_HOP_LEN / GSET_SAMPLE_RATE
GSET_PLAYERS = 6


class GuitarSetPPN(GuitarSet):
    """
    Simple wrapper over the GuitarSet Dataset, which extracts a pitch array with values in hertz 
    and a times array with timestamps in seconds.

    hop_length parameter is fixed to 256, sample_rate to 44100 as this makes the most sense with the Dataset. 
    Other values of hop_length require resampling of the pitch_array. 
    (TO DO: implement pitch array resampling)

    Keep in mind that 
    """
    def __init__(self, sample_rate: int=44100, seq_length: int=None, hop_length: int=None, **kwargs):
        """
        Initialize the dataset variant.

        Parameters
        ----------
        sample_rate (int)
            target sample rate, GuitarSet examples will be resampled into it

        seq_length (int)
            length of one audio frame, if None it will be deducted from 
            the sample rate and the number of target frames
            
        See GuitarSet class for others...
        """

        # Determine if the base directory argument was provided
        base_dir = kwargs.pop('base_dir', None)

        # Select a default base directory path if none was provided
        if base_dir is None:
            # Use the same naming scheme as regular GuitarSet
            base_dir = GSET_BASE_DIR

        if hop_length is None:
            hop_length = ppn.GSET_HOP_LEN

        # Update the argument in the collection
        kwargs.update({'base_dir' : base_dir})
        kwargs.update({'hop_length' : hop_length})
        kwargs.update({'sample_rate' : sample_rate if sample_rate is not None
                       else ppn.GSET_SAMPLE_RATE})

        super().__init__(**kwargs)

        # self.seq_length = seq_length


    def load(self, track):
        """
        Load the ground-truth from memory or generate it from scratch.

        Parameters
        ----------
        track : string
          Name of the track to load

        Returns
        ----------
        data : dict
          Dictionary with ground-truth for the track
        """
        
        # Load the track data if it exists in memory, otherwise instantiate track data
        data = TranscriptionDataset.load(self, track)

        # If the track data is being instantiated, it will not have the 'audio' key
        if not tools.query_dict(data, ppn.KEY_AUDIO):
            # Construct the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            audio, fs = tools.load_normalize_audio(wav_path,
                                                   fs=self.sample_rate,
                                                   norm=self.audio_norm)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(audio)

            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the original jams data
            jams_data = jams.load(jams_path)

            # Load the string-wise pitch annotations from the JAMS data
            pitch_array, times_array = self.extract_pitch_array_jams(jams_data, track)

            # Add evaluation extras to the dictionary
            data.update({
                ppn.KEY_FS : fs,
                ppn.KEY_AUDIO : audio,
                ppn.KEY_PITCH_ARRAY : pitch_array,
                ppn.KEY_TIMES : times_array
                })

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Create a copy of the data
                data_to_save = deepcopy(data)

                # Package the stacked pitch list into save-friendly format
                data_to_save.update({ppn.KEY_PITCH_ARRAY : pitch_array})

                data_to_save.update({ppn.KEY_TIMES : times_array})

                # Save the data as a NumPy zip file
                tools.save_dict_npz(gt_path, data_to_save)

        return data

    def get_track_data(self, track_id, sample_start=None, seq_length=None, snap_to_frame=True):
        """
        Get the features and ground truth for a track within a time interval.

        Parameters
        ----------
        track_id : string
          Name of track data to fetch
        sample_start : int
          Sample with which to begin the slice
        seq_length : int
          Number of samples to take for the slice
        snap_to_frame : bool
          Whether to begin exactly on frame boundaries or loose samples

        Returns
        ----------
        data : dict
          Dictionary with each entry sliced for the random or provided interval
        """

        if self.store_data:
            # Copy the track's ground-truth data into a local dictionary
            data = deepcopy(self.data[track_id])
        else:
            # Load the track's ground-truth
            data = self.load(track_id)

        if tools.KEY_FEATS not in data.keys():
            # Calculate the features and add to the dictionary
            data.update(self.calculate_feats(data))

        # Check to see if a specific sequence length was given
        if seq_length is None:
            # If not, and this Dataset object has a sequence length, use it
            if self.seq_length is not None:
                seq_length = self.seq_length
            # Otherwise, we assume the whole track is desired and perform no further actions
            else:
                return data

        # If a specific starting sample was not provided, sample one randomly
        if sample_start is None:
            sample_start = self.rng.randint(0, len(data[tools.KEY_AUDIO]) - seq_length + 1)

        # Determine the frames contained in this slice
        frame_start = sample_start // self.hop_length
        frame_end = frame_start + self.num_frames

        if snap_to_frame:
            # Snap the sample_start to the left-most frame boundary
            sample_start = frame_start * self.hop_length

        # Calculate the last sample included in the slice
        sample_end = sample_start + seq_length

        # Slice the audio
        data[tools.KEY_AUDIO] = data[tools.KEY_AUDIO][..., sample_start : sample_end + 1]

        # Determine the time in seconds of the boundary samples
        sec_start = sample_start / self.sample_rate
        sec_stop = sample_end / self.sample_rate

        if tools.query_dict(data, tools.KEY_NOTES):
            if isinstance(data[tools.KEY_NOTES], dict):
                # TODO - assumes stack consists of standard note groups
                # Extract the stacked notes and convert them to batched representations
                temp_stacked_notes = tools.apply_func_stacked_representation(data[tools.KEY_NOTES],
                                                                             tools.notes_to_batched_notes)
                # Perform time slicing w.r.t. the batched notes along each slice of the stack
                temp_stacked_notes = tools.apply_func_stacked_representation(temp_stacked_notes,
                                                                             tools.slice_batched_notes,
                                                                             start_time=sec_start,
                                                                             stop_time=sec_stop)
                # Convert back to standard note groups and update the dictionary
                data[tools.KEY_NOTES] = tools.apply_func_stacked_representation(temp_stacked_notes,
                                                                                tools.batched_notes_to_notes)
            else:
                # Slice the ground-truth notes if they exist in the ground-truth
                data[tools.KEY_NOTES] = tools.slice_batched_notes(data[tools.KEY_NOTES], sec_start, sec_stop)

        if tools.query_dict(data, tools.KEY_PITCHLIST):
            if isinstance(data[tools.KEY_PITCHLIST], dict):
                # Slice ground-truth pitch list by slice if exists in the ground-truth
                data[tools.KEY_PITCHLIST] = tools.apply_func_stacked_representation(data[tools.KEY_PITCHLIST],
                                                                                    tools.slice_pitch_list,
                                                                                    start_time=sec_start,
                                                                                    stop_time=sec_stop)
            else:
                # Slice ground-truth pitch list if exists in the ground-truth
                data[tools.KEY_PITCHLIST] = tools.slice_pitch_list(*data[tools.KEY_PITCHLIST], sec_start, sec_stop)

        # Define list of entries to skip during slicing process
        skipped_keys = [tools.KEY_AUDIO, tools.KEY_FS, tools.KEY_NOTES, tools.KEY_PITCHLIST]
        # Slice the remaining dictionary entries
        data = tools.slice_track(data, frame_start, frame_end, skipped_keys)

        return data

    @staticmethod
    def extract_pitch_array_jams(jam, track, uniform=True):
        """
        Extract pitch lists spread across slices (e.g. guitar strings) from JAMS annotations into a dictionary.

        Parameters
        ----------
        jam : JAMS object
          JAMS file data
        uniform : bool
          Whether to place annotations on a uniform time grid

        Returns
        ----------
        pitch_dict : dict
          Dictionary containing pitch_array with pitch values in Hz and time steps array
          pitch_array shape is (S, T), 
          time_steps array is of shape (T, )
          S - number of strings, T - number of time steps
        """
        # Extract all of the pitch annotations
        pitch_data_slices = jam.annotations[tools.constants.JAMS_PITCH_HZ]

        # Obtain the number of annotations
        stack_size = len(pitch_data_slices)

        # Initialize a dictionary to hold the pitch lists
        stacked_pitch_list = dict()
        slice_names = []

        # Loop through the slices of the stack
        for slc in range(stack_size):
            # Extract the pitch list pertaining to this slice
            slice_pitches = pitch_data_slices[slc]

            # Extract the string label for this slice
            string = slice_pitches.annotation_metadata[tools.constants.JAMS_STRING_IDX]
            slice_names.append(string)

            # Initialize an array/list to hold the times/frequencies associated with each observation
            entry_times, slice_pitch_list = np.empty(0), list()

            # Loop through the pitch observations pertaining to this slice
            for pitch in slice_pitches:
                # Extract the pitch
                freq = np.array([pitch.value['frequency']])

                # Don't keep track of zero or unvoiced frequencies
                if np.sum(freq) == 0 or not pitch.value['voiced']:
                    freq = np.empty(0)

                # Append the observation time
                entry_times = np.append(entry_times, pitch.time)
                # Append the frequency
                slice_pitch_list.append(freq)

            # Sort the pitch list before resampling just in case it is not already sorted
            entry_times, slice_pitch_list = tools.utils.sort_pitch_list(entry_times, slice_pitch_list)

            if uniform:
                # Align the pitch list with a uniform time grid
                entry_times, slice_pitch_list = tools.utils.time_series_to_uniform(
                        times=entry_times,
                        values=slice_pitch_list,
                        hop_length=ppn.GSET_TIME_STEP,
                        duration=jam.file_metadata.duration)

            # Add the pitch list to the stacked pitch list dictionary under the slice key
            stacked_pitch_list.update(tools.utils.pitch_list_to_stacked_pitch_list(entry_times, slice_pitch_list, string))

        # Determine the total number of observations in the uniform time series
        num_entries = int(np.ceil(jam.file_metadata.duration / ppn.GSET_TIME_STEP)) + 1
        time_steps_array = ppn.GSET_TIME_STEP * np.arange(num_entries)

        pitch_array_slices_list = []

        # for idx, slc in enumerate(slice_names):
        for slc in slice_names:
            # get the list of pitches in hz for slc string
            pitch_list = stacked_pitch_list[slc][1]

            # fill the empty numpy arrays in the pitch_list with zeros
            pitch_list = [np.zeros(1) if pitch.size == 0 else pitch for pitch in pitch_list]

            try: 
                # concatenate the whole thing into a numpy array
                pitch_list = np.concatenate(pitch_list)
            except ValueError:
                print(f"Empty array, track: {track}")
                print(f"Replacing with np.zeros({len(time_steps_array)})")
                pitch_list = np.zeros(len(time_steps_array))

            # append the slice to a list of all slices 
            pitch_array_slices_list.append(pitch_list)

        try: 
            pitch_array = np.vstack(pitch_array_slices_list)
        except ValueError as err:
            print(f"{err}, track: {track}, slice lengths: {[len(slice) for slice in pitch_array_slices_list]}")

        assert pitch_array.shape == (stack_size, time_steps_array.size)

        return pitch_array, time_steps_array
