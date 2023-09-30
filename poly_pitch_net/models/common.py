from amt_tools.models import TranscriptionModel, LogisticBank
from torch import nn


class CNNLogisticBank(LogisticBank):
    """
    Implements a multi-label logistic output layer designed to produce key activity,
    or more generally, quantized pitch activity with a CNN flavour instead of Linear.

    A straightforward example could correspond to a keyboard with 88 keys,
    where the output of each key is the sigmoid operation indicating whether
    or not the key is active.
    """

    def __init__(self, dim_in, dim_out, weights=None):
        """
        Initialize fields of the multi-label logistic layer.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        dim_out : int
          Dimensionality of output activations

        See OutputLayer class for others...
        """

        super().__init__(dim_in, dim_out, weights)

        # Initialize the output layer
        self.output_layer = nn.Conv1d(self.dim_in, self.dim_out, 1)
