# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from poly_pitch_net.tools.convert import bins_to_cents
import poly_pitch_net as ppn

# Regular imports
from torch import nn
import torch


class FretNetCrepe(nn.Module):
    """
    An improved model for discrete/continuous guitar tablature transcription,
    not dependent on the number of frames given.
    """

    def __init__(self, dim_in, in_channels, model_complexity=1,
                 matrix_path=None, device='cpu', frames=1,
                 no_strings=6, no_pitch_bins=360):
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
        self.no_strings = no_strings
        self.device = device

        # Initialize a flag to check whether to pad input features
        self.online = False

        # Number of filters for each convolutional block
        nf1 = 256   * model_complexity
        nf2 = 64 * model_complexity
        nf3 = 128 * model_complexity
        nf4 = 256 * model_complexity

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
            nn.Conv1d(nf4 * (self.dim_in // rd1[0] // rd2[0] // rd3[0]), self.no_strings*self.no_pitch_bins, 1),
            nn.Dropout(dpx),
            nn.Sigmoid()
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

        output[ppn.KEY_PITCH_LOGITS] = embeddings.unflatten(
                dim=1, sizes=(6, self.no_pitch_bins))
        # shape [B, 6, no_pitch_bins, T]

        return output

    def post_proc(self, output, pitch_names=None):
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
          shape [B, C, T]
        """
        offset = 4
        multi_pitch = output[ppn.KEY_PITCH_LOGITS]
        multi_pitch = multi_pitch.to(device=self.device)
        multi_pitch = multi_pitch.reshape(shape=(
            multi_pitch.shape[0],
            multi_pitch.shape[1],
            multi_pitch.shape[-1],
            multi_pitch.shape[2]
            ))
        # multi_pitch.shape == [B, C, T, O]

        # this has to be corrected
        if pitch_names is None:
            pitch_names = torch.arange(0, ppn.PITCH_BINS)
            pitch_names = pitch_names.expand(multi_pitch.shape)
            pitch_names = bins_to_cents(pitch_names)
        else:
            pitch_names = pitch_names.reshape(shape=multi_pitch.shape)

        pitch_names = pitch_names.to(device=self.device)

        # get argmax from each 360-vector
        centers = multi_pitch.argmax(dim=-1).to(torch.long)
        centers = centers.unsqueeze(-1)
        output[ppn.KEY_PITCH_CENTERS] = centers.squeeze(dim=-1)
        # centers.shape == [B, C, T]

        # weighted average: just the centers
        wg_avg = multi_pitch.gather(3, centers) * pitch_names.gather(3, centers)

        # weighted average: for offset in range [-4, -1], [1, 4]
        for off in range(1, offset):
            # left side of the offset range centers
            l_centers = centers - off
            l_multi_pitch = torch.where(
                    l_centers >= 0,
                    multi_pitch.gather(3, l_centers.where(l_centers >= 0, 0)),
                    0)
            l_pitch_names = torch.where(
                    l_centers >= 0,
                    pitch_names.gather(3, l_centers.where(l_centers >= 0, 0)),
                    0)
            wg_avg += l_multi_pitch * l_pitch_names

            # right side of the offset range centers
            r_centers = centers + off
            r_multi_pitch = torch.where(
                    r_centers < self.no_pitch_bins,
                    multi_pitch.gather(3, r_centers.where(r_centers < self.no_pitch_bins, 0)),
                    0)
            r_pitch_names = torch.where(
                    r_centers < self.no_pitch_bins,
                    pitch_names.gather(3, r_centers.where(r_centers < self.no_pitch_bins, 0)),
                    0)
            wg_avg += r_multi_pitch * r_pitch_names

        output[ppn.KEY_PITCH_WG_AVG] = wg_avg

        return output

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


class FretNetBlock(torch.nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pd: tuple[int, int] = (1, 1),
        dl: tuple[int, int] = (1, 1),
        ks: int =32
        ):

        layers = (
            nn.Conv2d(in_channels, out_channels, ks, padding=pd, dilation=dl),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, ks, padding=pd, dilation=dl),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        super().__init__(*layers)
