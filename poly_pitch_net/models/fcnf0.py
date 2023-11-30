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

    def __init__(self):
        layers = (
            Block(1, 256, 481, (2, 2)),
            Block(256, 32, 225, (2, 2)),
            Block(32, 32, 97, (2, 2)),
            Block(32, 128, 66),
            Block(128, 256, 35),
            Block(256, 512, 4),
            torch.nn.Conv1d(512, ppn.PITCH_BINS, 4))
        super().__init__(*layers)

    def pre_proc(self):
        pass

    def forward(self, frames):
        # shape=(batch, 1, penn.WINDOW_SIZE) =>
        # shape=(batch, penn.PITCH_BINS, penn.NUM_TRAINING_FRAMES)
        return super().forward(frames[:, :, 16:-15])

    def post_proc(self, frames):


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
