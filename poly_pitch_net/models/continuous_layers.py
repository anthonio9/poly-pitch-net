from guitar_transcription_continuous.models import L2LogisticBank
import torch as nn


class CNNL2LogisticBank(L2LogisticBank):
    """
    Implements a multi-label continuous-valued [0, 1] output layer with MSE loss.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize fields of the output layer.

        Parameters
        ----------
        See LogisticBank class...
        """
        super().__init__(dim_in, dim_out, None)
        self.output_layer = nn.Conv1d(dim_in, dim_out, 1)
