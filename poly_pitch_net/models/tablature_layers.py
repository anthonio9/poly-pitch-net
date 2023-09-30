from guitar_transcription_inhibition.models import LogisticTablatureEstimator
from amt_tools.models import LogisticBank
from torch import nn

from .common import CNNLogisticBank


class CNNLogisticTablatureEstimator(LogisticTablatureEstimator):
    """
    Multi-unit (string/fret) logistic tablature layer with pairwise inhibition
    with a *new school* CNN flavour.
    """
    def __init__(self, dim_in, profile, matrix_path=None, silence_activations=False, lmbda=1, device='cpu'):
        """
        Initialize a LogisticBank tablature layer and the inhibition matrix.

        Parameters
        ----------
        See TranscriptionModel class for others...

        matrix_path : str or None (optional)
          Path to inhibition matrix
        silence_activations : bool
          Whether to explicitly model silence
        lmbda : float
          Multiplier for the inhibition loss
        """

        super().__init__(dim_in, profile, device)

        self.silence_activations = silence_activations
        self.lmbda = lmbda

        # Extract tablature parameters
        num_strings = self.profile.get_num_dofs()
        num_pitches = self.profile.num_pitches

        # Calculate output dimensionality
        dim_out = num_strings * (num_pitches + int(self.silence_activations))

        # Set the tablature layer to a logistic bank
        self.tablature_layer = CNNLogisticBank(dim_in, dim_out)

        # Make sure the device is valid before creating the inhibition matrix
        self.change_device()

        if matrix_path is None:
            # Default the inhibition matrix if it does not exist (inhibit same-string pairs)
            self.inhibition_matrix = self.initialize_default_matrix(self.profile,
                                                                    self.silence_activations).to(self.device)
        else:
            # Load the inhibition matrix at the specified path
            self.set_inhibition_matrix(matrix_path)
