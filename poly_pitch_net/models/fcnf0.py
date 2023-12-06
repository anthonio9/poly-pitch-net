import poly_pitch_net as ppn
from poly_pitch_net.models import PitchNet, MonoPitchNet1D
import torch.nn as nn
import torch
# import torch.nn.functional as F
# from typing import Tuple
from copy import deepcopy


class FCNF0(MonoPitchNet1D):
    """Reimplementation of penn's FCNF0 interpretation: FCNF0++

    5 cents of bin width
    1440 pitch bins
    128 batch size
    8kHz sample rate
    1024 samples frame size ~ 128ms
    no normalization in FCNF0++
    """

    def __init__(self, no_pitch_bins: int = ppn.PITCH_BINS, string: int = 3):
        PitchNet.__init__(self)

        self.no_pitch_bins = no_pitch_bins
        self.string = string

        self.conv1 = FCNF0Block(1, 256, 481, (2, 2))
        self.conv2 = FCNF0Block(256, 32, 225, (2, 2))
        self.conv3 = FCNF0Block(32, 32, 97, (2, 2))
        self.conv4 = FCNF0Block(32, 128, 66)
        self.conv5 = FCNF0Block(128, 256, 35)
        self.conv6 = FCNF0Block(256, 512, 4)
        self.pitch_head = torch.nn.Conv1d(512, no_pitch_bins, 4)

    def pre_proc(self, input):
        assert input[ppn.KEY_AUDIO].shape[-1] == 1024

        #output = {}
        #output[ppn.KEY_AUDIO] = deepcopy(input[ppn.KEY_AUDIO])
        #output[ppn.KEY_PITCH_ARRAY] = deepcopy(input[ppn.KEY_PITCH_ARRAY])

        ## transform [batch_size, samples] ->
        ## [batch_size, 1, samples]
        #output[ppn.KEY_AUDIO] = output[ppn.KEY_AUDIO][:, None, :]
        #output[ppn.KEY_PITCH_ARRAY] = output[ppn.KEY_PITCH_ARRAY][:, self.string, -1]
        #output[ppn.KEY_PITCH_ARRAY] = output[ppn.KEY_PITCH_ARRAY][:, None]

        input[ppn.KEY_AUDIO] = input[ppn.KEY_AUDIO][:, None, :]
        input[ppn.KEY_PITCH_ARRAY] = input[ppn.KEY_PITCH_ARRAY][:, self.string, -1]
        input[ppn.KEY_PITCH_ARRAY] = input[ppn.KEY_PITCH_ARRAY][:, None]

        #return output
        return input 

    def forward(self, input):
        input = self.pre_proc(input)
        frames = input[ppn.KEY_AUDIO]
        frames = frames.to(self.device)

        embeddings = self.conv1(frames[:, :, 16:-15])
        embeddings = self.conv2(embeddings)
        embeddings = self.conv3(embeddings)
        embeddings = self.conv4(embeddings)
        embeddings = self.conv5(embeddings)
        embeddings = self.conv6(embeddings)
        embeddings = self.pitch_head(embeddings)

        output = {}
        output[ppn.KEY_PITCH_LOGITS] = embeddings

        return output


class FCNF0Block(nn.Sequential):

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
            layers += (torch.nn.Dropout(ppn.DROPOUT),)

        super().__init__(*layers)
