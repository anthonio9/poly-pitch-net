import torch

import poly_pitch_net as ppn


def poly_pitch_loss(logits, pitch):
    # reshape [B, C, O, T] ==> [B, C, T, O]
    logits = logits.reshape(shape=(
        logits.shape[0],
        logits.shape[1],
        logits.shape[-1],
        logits.shape[2]
        ))

    if pitch_names is None:
        no_pitch_bins = ppn.PITCH_BINS
    else:
        # keep compatibility with the tests
        no_pitch_bins = pitch_names.shape[-1]
        
    logits = logits.reshape(-1, no_pitch_bins).float()
    pitch = pitch.flatten()

    loss.cents = ppn.tools.convert.bins_to_cents(
            torch.arange(no_pitch_bins))[:, None]

    # Ensure values are on correct device (no-op if devices are the same)
    loss.cents = loss.cents.to(pitch.device)

    # Create normal distributions
    distributions = torch.distributions.Normal(
            ppn.tools.convert.frequency_to_cents(pitch), 
            25)

    # Sample normal distributions
    pitch_bins = torch.exp(distributions.log_prob(loss.cents)).permute(1, 0)

    # Normalize
    pitch_bins = pitch_bins / (pitch_bins.max(dim=1, keepdims=True).values + 1e-8)
    
    # Compute binary cross-entropy loss
    return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            pitch_bins)


def onehot_with_ignore_label(labels, num_class, ignore_label):
    dummy_label = num_class + 1

    # set the mask for the ingored labels
    mask = labels == ignore_label
    modified_labels = labels.clone()

    # set ignored labels to max value
    modified_labels[mask] = num_class

    # one-hot encode the modified labels
    one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=dummy_label)

    # remove the last row in the one-hot encoding
    one_hot_labels = one_hot_labels[:, :, :-1]
    return one_hot_labels


def onehot_with_silence(labels, num_class, silence_label=torch.tensor(0)):
    labels_with_silence = num_class + 1

    # set the mask for the silence labels
    mask = labels == silence_lable
    modified_labels = labels.clone()

    # set silence labels to max value
    modified_labels[mask] = num_class

    # one-hot encode the modified labels
    one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=labels_with_silence)

    # remove the last row in the one-hot encoding
    one_hot_labels = one_hot_labels[:, :, :-1]
    return one_hot_labels


def mono_pitch_loss(logits, pitch):
    # transform logits of shape [B, O, T] into [B, T, O]
    logits = logits.permute(0, 2, 1).reshape(-1, ppn.PITCH_BINS)

    # start with a simple pitch_bins vector, make it one-hot
    pitch_bins = ppn.tools.frequency_to_bins(pitch)
    pitch_bins_1hot = onehot_with_ignore_label(pitch_bins, ppn.PITCH_BINS, torch.tensor(0))

    pitch_bins_1hot = pitch_bins_1hot.float()
    pitch_bins_1hot = pitch_bins_1hot.reshape(-1, ppn.PITCH_BINS)
    
    # Compute binary cross-entropy loss
    return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            pitch_bins_1hot)


def loss(model, logits, pitch, pitch_names=None):
    """
    Compute the CREPE loss for 6 channels

    Parameters
    ----------
    logits - output from the FretNetCrepe.forward() function.
    Should be of shape [B, C, O, T]

    pitch - ground truth, values of the true pitch in Hz

    look here for more 
    https://github.com/interactiveaudiolab/penn/blob/955656618bb71e1e2f040d1d134b8de834f51733/penn/train/core.py
    """
    # breakpoint()

    if 'FretNetCrepe' in model.model_name():
        return poly_pitch_loss(logits, pitch)
    
    if 'MonoPitchNet1D' in model.model_name():
        return mono_pitch_loss(logits, pitch)
