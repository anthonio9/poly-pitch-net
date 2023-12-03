import poly_pitch_net as ppn
from poly_pitch_net.models import PitchNet
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple


class FCNF0(torch.nn.Sequential):
    """Reimplementation of penn's FCNF0 interpretation: FCNF0++
    
    5 cents of bin width
    1440 pitch bins
    128 batch size
    8kHz sample rate
    1024 samples frame size ~ 128ms
    no normalization in FCNF0++
    """

    def __init__(self, no_pitch_bins: int=ppn.PITCH_BINS, string: int=3):
        self.no_pitch_bins = no_pitch_bins
        self.string = string

        layers = (
            FCNF0Block(1, 256, 481, (2, 2)),
            FCNF0Block(256, 32, 225, (2, 2)),
            FCNF0Block(32, 32, 97, (2, 2)),
            FCNF0Block(32, 128, 66),
            FCNF0Block(128, 256, 35),
            FCNF0Block(256, 512, 4),
            torch.nn.Conv1d(512, no_pitch_bins, 4))
        super().__init__(*layers)

    def pre_proc(self, input):
        assert input[ppn.KEY_AUDIO].shape[-1] == 1024

        # transform [batch_size, samples] -> 
        # [batch_size, 1, samples]
        input[ppn.KEY_AUDIO] = input[ppn.KEY_AUDIO][:, None, :]

        return input

    def forward(self, input):
        input = self.pre_proc(input)
        frames = input[ppn.KEY_AUDIO]

        output = {}
        output[ppn.KEY_PITCH_LOGITS] = super().forward(frames[:, :, 16:-15])

        return output

    def post_proc(self, frames):
        pass


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
