# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
import guitar_transcription_continuous.utils as utils
import amt_tools.tools as tools

# Regular imports
from torch import nn
import math


class FretNetCrepe(nn.Module):
    """
    An improved model for discrete/continuous guitar tablature transcription,
    not dependent on the number of frames given.
    """

    def __init__(self, dim_in, in_channels, model_complexity=1,
                 matrix_path=None, device='cpu', frames=1, no_pitch_bins=360):
        """
        Initialize all components of the model.

        Parameters
        ----------
        See TabCNNLogisticContinuous/LogisticTablatureEstimator class for others...

        cont_layer : bool
          Switch to select type of continuous output layer for relative pitch prediction
          (0 - Continuous Bernoulli | 1 - MSE | None - disable relative pitch prediction)
        """
        
        nn.Module.__init__(self)

        self.no_pitch_bins = no_pitch_bins
        self.in_channels = in_channels
        self.dim_in = dim_in

        # Initialize a flag to check whether to pad input features
        self.online = False

        # Number of filters for each convolutional block
        nf1 = 16 * model_complexity
        nf2 = 32 * model_complexity
        nf3 = 48 * model_complexity
        nf4 = 64 * model_complexity

        # Kernel size for each convolutional block
        ks1 = (3, 3)
        ks2 = (3, 3)
        ks3 = ks2
        ks4 = ks3
        ks5 = ks4

        # Padding amount for each convolutional block
        pd1 = (1, 1)
        pd2 = (1, 1)
        pd3 = pd2
        pd4 = pd3
        pd5 = pd4

        # Reduction size for each pooling operation
        rd1 = (2, 1)
        rd2 = rd1
        rd3 = rd2

        # Dropout percentages for each dropout operation
        dp1 = 0.5
        dp2 = 0.25
        dpx = 0.10

        # Dilation
        dl1 = (1, 1)
        dl2 = (1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, nf1, ks1, padding=pd1, dilation=dl1),
            nn.BatchNorm2d(nf1),
            nn.ReLU(),
            nn.Conv2d(nf1, nf1, ks2, padding=pd1, dilation=dl1),
            nn.BatchNorm2d(nf1),
            nn.ReLU(),
        )
        # shape [B, nf1, F, T]

        self.conv2 = nn.Sequential(
            nn.Conv2d(nf1, nf2, ks2, padding=pd2, dilation=dl2),
            nn.BatchNorm2d(nf2),
            nn.ReLU(),
            nn.Conv2d(nf2, nf2, ks2, padding=pd2),
            nn.BatchNorm2d(nf2),
            nn.ReLU()
        )
        # shape [B, nf2, F, T]

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(rd1),
            nn.Dropout(dp1)
        )
        # shape [B, nf2, F/2, T]

        self.conv3 = nn.Sequential(
            nn.Conv2d(nf2, nf3, ks3, padding=pd3),
            nn.BatchNorm2d(nf3),
            nn.ReLU(),
            nn.Conv2d(nf3, nf3, ks3, padding=pd3),
            nn.BatchNorm2d(nf3),
            nn.ReLU()
        )
        # shape [B, nf3, F/2, T]

        self.pool2 = nn.Sequential(
            nn.MaxPool2d(rd2),
            nn.Dropout(dp2)
        )
        # shape [B, nf3, F/4, T]

        self.conv4 = nn.Sequential(
            nn.Conv2d(nf3, nf4, ks4, padding=pd4),
            nn.BatchNorm2d(nf4),
            nn.ReLU(),
            nn.Conv2d(nf4, nf4, ks4, padding=pd4),
            nn.BatchNorm2d(nf4),
            nn.ReLU()
        )
        # shape [B, nf4, F/4, T]

        self.pool3 = nn.Sequential(
            nn.MaxPool2d(rd3),
            nn.Dropout(dpx),
            # nn.Flatten(start_dim=1, end_dim=2)
        )

        self.pitch_head = nn.Sequential(
            nn.Conv1d(nf4 * (self.dim_in // rd1[0] // rd2[0] // rd3[0]), 6*360, 1),
            nn.Dropout(dpx),
        )

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

    def forward(self, feats):
        """
        Perform the main processing steps for FretNet.

        Parameters
        ----------
        feats : Tensor (B x T x C x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          C - number of channels in features
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict with pitch and (all B x T x O)
          Dictionary containing continuous pitch output
          B - batch size,
          C - number of channels
          O - number of output neurons (dim_out)
          T - number of time steps (frames),
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the batch size before sequence-frame axis is collapsed
        batch_size = feats.size(0)

        # Collapse the sequence-frame axis into the batch axis as in TabCNN implementation
        # feats = feats.reshape(-1, self.in_channels, self.dim_in, self.frame_width)
        # input shape is [B, T, C, F, 1], change it to [B, C, F, T]
        feats = feats.reshape(batch_size, self.in_channels, self.dim_in, -1)

        # Obtain the feature embeddings from the model
        embeddings = self.conv1(feats)
        # shape [B, nf1, F, T]

        embeddings = self.conv2(embeddings)
        # shape [B, nf2, F, T]

        embeddings = self.pool1(embeddings)
        # shape [B, nf2, F/2, T]

        embeddings = self.conv3(embeddings)
        # shape [B, nf3, F/2, T]

        embeddings = self.pool2(embeddings)
        # shape [B, nf3, F/4, T]

        embeddings = self.conv4(embeddings)
        # shape [B, nf4, F/4, T]

        embeddings = self.pool3(embeddings)
        # shape [B, nf4, F/8, T]

        embeddings = embeddings.flatten(start_dim=1, end_dim=2)
        # shape [B, nf4*F/8, T]

        embeddings = self.pitch_head(embeddings)
        # shape [B, 6*no_pitch_bins, T]

        output[tools.KEY_MULTIPITCH] = embeddings.unflatten(dim=1, sizes=(6, self.no_pitch_bins))
        # shape [B, 6, no_pitch_bins, T]

        return output

    def post_proc(self, output):
        """
        Calculate final weight averaged pitch value

        Parameters
        ----------
        output : dict
          Dictionary including model output and time vector.
          "pitch"
          B - batch size,
          C - number of channels,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)

          "time"
          B - batch size,
          T - number of time steps (frames)


        Returns
        ----------
        output : list of tuples
          List containing time / pitch bindings.
        """

        # get argmax from each 360-vector
        centers = output[tools.KEY_MULTIPITCH].argmax(dim=-1)

        # do [center - 4, center + 4] averaging
        output = centers

        return output

