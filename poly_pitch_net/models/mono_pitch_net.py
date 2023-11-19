import poly_pitch_net as ppn
from poly_pitch_net.models import PitchNet
import penn
import torch.nn as nn
import torch


class MonoPitchNet1D(PitchNet):
    """A small model focused on guitar pitch recognition. 

    The purpose of the model is to recognize pitch from one string only, 
    no matter the amount of strings taking part in the provided data. 

    The main aim is to use this model as means of understanding the issues with the larger 
    PolyPitchNet model. First undrestand, then fix them.

    MonoPitchNet is expecting only one block of CQT / STFT per time step.
    """

    def __init__(self, dim_in: int, no_pitch_bins: int=360,
                 register_silence: bool=False, string=3):
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
        self.string = string

        self.conv1 = MonoPitchBlock1D(self.dim_in, 256)
        self.conv2 = MonoPitchBlock1D(256, 32)
        self.conv3 = MonoPitchBlock1D(32, 32)
        self.conv4 = MonoPitchBlock1D(32, 128)
        self.conv5 = MonoPitchBlock1D(128, 256)
        self.conv6 = MonoPitchBlock1D(256, 512)

        self.pitch_head = nn.Conv1d(
                512,
                no_pitch_bins + int(register_silence), 1)

    def pre_proc(self, input: dict):
        # choose HCQT channel 0
        assert len(input[ppn.KEY_FEATURES].shape) == 4
        input[ppn.KEY_FEATURES] = input[ppn.KEY_FEATURES][:, 0, :, :]

        # choose string 3
        assert len(input[ppn.KEY_PITCH_ARRAY].shape) == 3
        input[ppn.KEY_PITCH_ARRAY] = input[ppn.KEY_PITCH_ARRAY][:, self.string, :]
        
        return input

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
        inpput = self.pre_proc(input)
        features = input[ppn.KEY_FEATURES]

        # always be sure about the right device
        features = features.to(self.device)

        embeddings = self.conv1(features)
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

        # zero_max_bins = (pitch_bins == 0).sum()
        # print(f"Zero max bins: {zero_max_bins}")

        pitch_cents = ppn.tools.convert.bins_to_cents(
                pitch_bins,
                register_silence=self.register_silence)
        pitch_hz = ppn.tools.convert.bins_to_frequency(
                pitch_bins,
                register_silence=self.register_silence)
        input[ppn.KEY_PITCH_ARRAY_CENTS] = pitch_cents
        input[ppn.KEY_PITCH_ARRAY_HZ] = pitch_hz

        return input


class MonoPitchNetTime(PitchNet):
    """Mono Pitch recognition model based of penn FCNF0"""

    def __init__(self, dim_in: int, no_pitch_bins: int=360,
                 register_silence: bool=False, string=3):
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
        self.string = string

        layers = (penn.model.Normalize(),) if ppn.NORMALIZE_INPUT else ()

        layers += (
            MonoPitchBlockTime(1, 256, 223, kernel_size=64),
            MonoPitchBlockTime(256, 32, 112, kernel_size=32),
            MonoPitchBlockTime(32, 32, 97, kernel_size=32),
            MonoPitchBlockTime(32, 128, 66),
            MonoPitchBlockTime(128, 256, 35),
            MonoPitchBlockTime(256, 512, 4),
            torch.nn.Conv1d(512, penn.PITCH_BINS + int(self.register_silence), 4)
                )

        self.sequence = torch.nn.Sequential(*layers)


    def pre_proc(self, input):
        # choose the string
        assert len(input[ppn.KEY_PITCH_ARRAY].shape) == 3
        input[ppn.KEY_PITCH_ARRAY] = input[ppn.KEY_PITCH_ARRAY][:, self.string, :]

        chunks = input[ppn.KEY_PITCH_ARRAY].shape[-1]
        features = input[ppn.KEY_AUDIO]
        pad_right = chunks * ppn.GSET_HOP_LEN - features.shape[-1]

        # padd up to a nice value of 200 * 256 frames
        features = nn.functional.pad(features, (0, pad_right), 'constant', 0)
        
        features_chunked = features.chunk(chunks=chunks, dim=-1)
        features = torch.stack(features_chunked, dim=-1)

        assert len(features.shape) == 3

        #features = features[:, None, :, :]

        input[ppn.KEY_AUDIO] = features

        return input

    def forward(self, input: dict):
        input = self.pre_proc(input)

        breakpoint()

        features = input[ppn.KEY_AUDIO].to(self.device)

        output = {}
        output[ppn.KEY_PITCH_LOGITS] = self.sequence(features)

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
        zero_max_bins = (pitch_bins == 0).sum()
        print(f"Zero max bins: {zero_max_bins}")

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


class MonoPitchBlockTime(torch.nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        pooling=None,
        kernel_size=32):
        layers = (
            torch.nn.Conv1d(in_channels, out_channels, kernel_size),
            torch.nn.ReLU())

        # Maybe add pooling
        if pooling is not None:
            layers += (torch.nn.MaxPool1d(*pooling),)

        # Maybe add normalization
        if ppn.NORMALIZATION == 'batch':
            layers += (torch.nn.BatchNorm1d(out_channels, momentum=.01),)
        elif ppn.NORMALIZATION == 'instance':
            layers += (torch.nn.InstanceNorm1d(out_channels),)
        elif ppn.NORMALIZATION == 'layer':
            layers += (torch.nn.LayerNorm((out_channels, length)),)
        else:
            raise ValueError(
                f'Normalization method {ppn.NORMALIZATION} is not defined')

        # Maybe add dropout
        if ppn.DROPOUT is not None:
            layers += (torch.nn.Dropout(penn.DROPOUT),)

        super().__init__(*layers)
