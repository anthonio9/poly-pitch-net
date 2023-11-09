import poly_pitch_net as ppn
import torch.nn as nn


class MonoPitchNet(nn.module):
    """A small model focused on guitar pitch recognition. 

    The purpose of the model is to recognize pitch from one string only, 
    no matter the amount of strings taking part in the provided data. 

    The main aim is to use this model as means of understanding the issues with the larger 
    PolyPitchNet model. First undrestand, then fix them.
    """

    def __init__(self, dim_in: int, in_channels: int, no_pitch_bins: int=360):
        """Initialize all components of MonoPitchNet model.

        Args:
            dim_in (int): number of frequency bins per block of the provided input data.
            in_channels (int): number of the input channels of the first conv2d layer.
            no_pitch_bins (int): number of the output pitch bins logits, defaults to 360.
        """

        self.dim_int = dim_in
        self.in_channels = in_channels
        self.no_pitch_bins = no_pitch_bins

        self.conv1 = MonoPitchBlock(in_channels, 256)
        self.conv2 = MonoPitchBlock(256, 32, reduction=2)
        self.conv3 = MonoPitchBlock(32, 128, reduction=2)

        self.pitch_head = nn.Conv1d(
                128 * dim_in // self.conv2.reduction // self.conv3.reduction,
                no_pitch_bins, 1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.pitch_head(output)

        return output


class MonoPitchBlock(nn.Sequential):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple[int, int]=(3, 3),
            padding: tuple[int, int]=(1, 1),
            dilation: tuple[int, int]=(1, 1),
            reduction: int=None
            ):
        layers = (
                nn.Conv2d(in_channels,
                          out_channels, 
                          kernel_size=kernel_size,
                          padding=padding,
                          dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )

        self.reduction = 1

        if reduction is not None:
            self.reduction = reduction
            layers += (
                    nn.MaxPool2d(self.reduction)
                    nn.Dropout((1, 1))
                    )

        super().__init__(*layers)
