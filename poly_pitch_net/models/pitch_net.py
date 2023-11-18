import torch.nn as nn
from abc import abstractmethod


class PitchNet(nn.Module):
    """Base abstract class for pitch recognition models"""

    def __init__(self):
        nn.Module.__init__(self)

    def model_name(cls):
        """
        Retrieve an appropriate tag, the class name, for the model.

        Returns
        ----------
        tag : str
          Name of the child class calling the function
        """

        tag = cls.__name__

        return tag

    def change_device(self, device=None):
        """
        Change the device and load the model onto the new device.

        Parameters
        ----------
        device : string, int or None, optional (default None)
          Device to load model onto
        """

        if device is None:
            # If the function is called without a device, use the current device
            device = self.device

        if isinstance(device, int):
            # If device is an integer, assume device represents GPU number
            device = torch.device(f'cuda:{device}'
                                  if torch.cuda.is_available() else 'cpu')

        # Change device field
        self.device = device
        # Load the transcription model onto the device
        self.to(self.device)

    @abstractmethod
    def pre_proc(self, input: dict):
        """Process data to be suited for the model

        Args:
            input (dict)
                batch dict returned by GuitarSet lodaer

        Returns:
            output (dict)
                batch dict with adjusted input data
        """
        pass

    @abstractmethod
    def froward(self, input: dict):
        pass

    @abstractmethod
    def post_proc(self, input: dict):
        """Process logits into pitch information

        Args:
            input (dict)
                output returned from the forward function

        Returns:
            output (dict)
                input dict with new pitch members
        """
        pass
