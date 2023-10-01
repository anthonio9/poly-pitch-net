# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import TranscriptionModel
from guitar_transcription_continuous.models import TabCNNLogisticContinuous
from guitar_transcription_continuous.models import CBernoulliBank
from .tablature_layers import CNNLogisticTablatureEstimator
from .continuous_layers import CNNL2LogisticBank
from .common import CNNLogisticBank

import guitar_transcription_continuous.utils as utils

import amt_tools.tools as tools

# Regular imports
from torch import nn

import math


class FretNetCNN(TabCNNLogisticContinuous):
    """
    An improved model for discrete/continuous guitar tablature transcription,
    not dependent on the number of frames given.
    """

    def __init__(self, dim_in, profile, in_channels, model_complexity=1, semitone_radius=0.5,
                 gamma=1, cont_layer=1, matrix_path=None, silence_activations=False, lmbda=1,
                 estimate_onsets=True, device='cpu', frames=9):
        """
        Initialize all components of the model.

        Parameters
        ----------
        See TabCNNLogisticContinuous/LogisticTablatureEstimator class for others...

        cont_layer : bool
          Switch to select type of continuous output layer for relative pitch prediction
          (0 - Continuous Bernoulli | 1 - MSE | None - disable relative pitch prediction)
        estimate_onsets : bool
          Switch for including an additional head to estimate onsets
        """

        TranscriptionModel.__init__(self, dim_in, profile, in_channels,
                                    model_complexity, frames, device)

        self.semitone_radius = semitone_radius
        self.gamma = gamma
        self.cont_layer = cont_layer
        self.estimate_onsets = estimate_onsets

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
            nn.Flatten(start_dim=1, end_dim=2)
        )
        # shape [B, nf4*F/8, T]

        # could this be the tablature_head?
        # self.conv_final = nn.Sequential(
        #     nn.Conv1d(nf4 * dim_in / rd1[0] / rd2[0] / rd3[0], 126, 1),
        #     nn.BatchNorm1d(126),
        #     nn.ReLU(),
        #     nn.Dropout(dpx),
        #     nn.Unflatten(2, (6, 21))
        # )
        # shape [B, 6, 21, T]

        def pooling_reduction(dim_in, times=1):
            # Define a simple recursive function to compute dimensionality after all pooling operations
            return dim_in if times <= 0 else pooling_reduction(math.ceil(dim_in / 2), times - 1)

        # Compute the dimensionality of feature embeddings
        features_dim_in = nf4 * pooling_reduction(dim_in, times=3)
        # Reduce the dimensionality by half before feeding to output layers
        features_dim_int = features_dim_in // 2

        # Initialize a logistic output layer for discrete tablature estimation
        self.tablature_layer = CNNLogisticTablatureEstimator(
                dim_in=features_dim_int,
                profile=profile,
                matrix_path=matrix_path,
                silence_activations=silence_activations,
                lmbda=lmbda,
                device=device)

        # Initialize the discrete tablature estimation head
        self.tablature_head = nn.Sequential(
            nn.Conv1d(features_dim_in, features_dim_int, 1),
            nn.ReLU(),
            nn.Dropout(dpx),
            self.tablature_layer
            # this returns 
        )

        # Determine output dimensionality when not explicitly modeling silence
        dim_out = self.profile.get_num_dofs() * self.profile.num_pitches

        if self.cont_layer is not None:
            # Create another output layer to estimate relative pitch deviation
            if self.cont_layer:
                # Train continuous relative pitch layer with MSE loss
                self.relative_layer = CNNL2LogisticBank(features_dim_int, dim_out)
            else:
                # Train continuous relative pitch layer with Continuous Bernoulli loss
                self.relative_layer = CBernoulliBank(features_dim_int, dim_out)

            # Initialize the relative tablature estimation head
            self.relative_head = nn.Sequential(
                nn.Conv1d(features_dim_in, features_dim_int, 1),
                nn.ReLU(),
                nn.Dropout(dpx),
                self.relative_layer
            )

        if self.estimate_onsets:
            # Initialize an output layer for onset detection
            self.onsets_layer = CNNLogisticBank(features_dim_int, dim_out)

            # Initialize the onset detection head
            self.onsets_head = nn.Sequential(
                nn.Conv1d(features_dim_in, features_dim_int, 1),
                nn.ReLU(),
                nn.Dropout(dpx),
                self.onsets_layer
            )

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
        output : dict w/ tablature/relative and potentially onsets Tensors (all B x T x O)
          Dictionary containing continuous tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the batch size before sequence-frame axis is collapsed
        batch_size = feats.size(0)

        # Collapse the sequence-frame axis into the batch axis as in TabCNN implementation
        feats = feats.reshape(-1, self.in_channels, self.dim_in, self.frame_width)

        # Obtain the feature embeddings from the model
        embeddings = self.conv1(feats)
        # print(f"conv1 shape: {embeddings.shape}")

        embeddings = self.conv2(embeddings)
        # print(f"conv2 shape: {embeddings.shape}")

        embeddings = self.pool1(embeddings)
        embeddings = self.conv3(embeddings)
        # print(f"conv3 shape: {embeddings.shape}")

        embeddings = self.pool2(embeddings)
        embeddings = self.conv4(embeddings)

        embeddings = self.pool3(embeddings)

        # Flatten spatial features into one embedding
        #embeddings = embeddings.flatten(1)
        # Size of the embedding
        #embedding_size = embeddings.size(-1)
        # Restore proper batch dimension, unsqueezing sequence-frame axis
        #embeddings = embeddings.view(batch_size, -1, embedding_size)

        # Process embeddings with discrete tablature head
        output[tools.KEY_TABLATURE] = self.tablature_head(embeddings).pop(tools.KEY_TABLATURE)

        if self.cont_layer is not None:
            # Process embeddings with relative tablature head
            output[utils.KEY_TABLATURE_REL] = self.relative_head(embeddings)

        if self.estimate_onsets:
            # Process embeddings with onsets head
            output[tools.KEY_ONSETS] = self.onsets_head(embeddings)

        return output

    def post_proc(self, batch):
        """
        Calculate onsets loss and finalize onset predictions.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing tablature, relative deviation, and potentially onsets/loss
        """

        # Call the post-processing method of the parent
        output = super().post_proc(batch)

        if self.estimate_onsets:
            # Obtain the estimated onsets
            onsets_est = output[tools.KEY_ONSETS]

            # Unpack the loss if it exists
            loss = tools.unpack_dict(output, tools.KEY_LOSS)
            total_loss = 0 if loss is None else loss[tools.KEY_LOSS_TOTAL]

            if loss is None:
                # Create a new dictionary to hold the loss
                loss = {}

            # Check to see if ground-truth onsets are available
            if tools.KEY_ONSETS in batch.keys():
                # Extract the ground-truth onsets
                onsets_ref = tools.stacked_multi_pitch_to_logistic(batch[tools.KEY_ONSETS], self.profile, False)
                # Calculate the onsets loss term
                onsets_loss = self.onsets_layer.get_loss(onsets_est, onsets_ref)
                # Add the onsets loss to the tracked loss dictionary
                loss[tools.KEY_LOSS_ONSETS] = onsets_loss
                # Add the (potentially) scaled-down onsets loss to the total loss
                total_loss += ((1 / self.gamma) ** int(self.cont_layer == 0)) * onsets_loss

            # Determine if loss is being tracked
            if total_loss:
                # Add the loss to the output dictionary
                loss[tools.KEY_LOSS_TOTAL] = total_loss
                output[tools.KEY_LOSS] = loss

            # Finalize the predicted onsets
            onsets_est = self.onsets_layer.finalize_output(onsets_est, 0.5)
            # Convert the onsets to stacked multi pitch format
            output[tools.KEY_ONSETS] = tools.logistic_to_stacked_multi_pitch(onsets_est, self.profile, False)

        return output

