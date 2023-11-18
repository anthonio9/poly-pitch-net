import poly_pitch_net as ppn
import torch.nn as nn
import torch


class MonoPitchNet1D(nn.Module):
    """A small model focused on guitar pitch recognition. 

    The purpose of the model is to recognize pitch from one string only, 
    no matter the amount of strings taking part in the provided data. 

    The main aim is to use this model as means of understanding the issues with the larger 
    PolyPitchNet model. First undrestand, then fix them.

    MonoPitchNet is expecting only one block of CQT / STFT per time step.
    """

    def __init__(self, dim_in: int, no_pitch_bins: int=360,
                 register_silence: bool=False):
        """Initialize all components of MonoPitchNet model.

        Args:
            dim_in (int): number of frequency bins per block of the provided input data,
            this is also the number of channels in the first Conv1d layer.
            no_pitch_bins (int): number of the output pitch bins logits, defaults to 360.
            register_silence (bool): register_silence (True) with the model or not (False)
        """

        nn.Module.__init__(self)

        self.dim_in = dim_in
        self.no_pitch_bins = no_pitch_bins
        self.register_silence = register_silence

        self.conv1 = MonoPitchBlock1D(self.dim_in, 256)
        self.conv2 = MonoPitchBlock1D(256, 32)
        self.conv3 = MonoPitchBlock1D(32, 32)
        self.conv4 = MonoPitchBlock1D(32, 128)
        self.conv5 = MonoPitchBlock1D(128, 256)
        self.conv6 = MonoPitchBlock1D(256, 512)

        self.pitch_head = nn.Conv1d(
                512,
                no_pitch_bins + int(register_silence), 1)

    def forward(self, input):
        """Process data and input pitch logits.

        Run through the Conv1d layers of pitch ned, to finally pass through
        the pitch head which output pitch logits. 
        
        Args:
            input (tensor [B, C, T]) - CQT / STFT blocks,

        Returns:
            output (tensor [B, O, T]) - pitch logits

        B - batch size,
        T - number of the time frames
        C - number of the CQT / STFT bins, given in the init function
        O - number of pitch bins, given in the init function
        """
        embeddings = self.conv1(input)
        embeddings = self.conv2(embeddings)
        embeddings = self.conv3(embeddings)
        embeddings = self.conv4(embeddings)
        embeddings = self.conv5(embeddings)
        embeddings = self.conv6(embeddings)
        embeddings = self.pitch_head(embeddings)

        # Initialize an empty dictionary to hold output
        output = dict()

        output[ppn.KEY_PITCH_LOGITS] = embeddings

        return output

    def post_proc(self, input: dict):
        """Process logits into cents using argmax

        Args:
            input (dict)
                output returned from the forward function
        """
        logits = input[ppn.KEY_PITCH_LOGITS]
        # reshape [B, O, T] into [B, T, O]
        logits = logits.permute(0, 2, 1)
        logits = torch.nn.functional.softmax(logits, dim=-1)

        assert torch.all(logits.isfinite())

        # this should be of shape [B, T]
        pitch_bins = logits.argmax(dim=-1)
        assert pitch_bins.shape == logits.shape[:-1]

        # count how many argmaxs are equal to 0
        # zero_max_bins = (pitch_bins == 0).count_nonzero()
        # print(f"Zero max bins: {zero_max_bins}")

        # if self.register_silence:
        #    pitch_bins[logits[:, :, 360] > 0.5] = ppn.PITCH_BINS

        pitch_cents = ppn.tools.convert.bins_to_cents(
                pitch_bins,
                register_silence=self.register_silence)
        pitch_hz = ppn.tools.convert.bins_to_frequency(
                pitch_bins,
                register_silence=self.register_silence)
        input[ppn.KEY_PITCH_ARRAY_CENTS] = pitch_cents
        input[ppn.KEY_PITCH_ARRAY_HZ] = pitch_hz

        return input

    @classmethod
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



class MonoPitchBlock1D(nn.Sequential):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=5,
            padding: int=2,
            dilation: int=1,
            ):
        layers = (
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels, 
                          kernel_size=kernel_size,
                          padding=padding,
                          dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
                )

        super().__init__(*layers)
