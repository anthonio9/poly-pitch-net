import torch

import poly_pitch_net as ppn


def poly_pitch_loss(logits, pitch, register_silence=False):
    # number of all bins, in / ex cluding silence
    no_all_bins = ppn.PITCH_BINS + int(register_silence)

    # transform logits of shape [B, O, T] into [B, T, O]
    logits = logits.permute(0, 2, 1).reshape(-1, no_all_bins)

    pitch = pitch.flatten()

    loss.cents = ppn.tools.convert.bins_to_cents(
            torch.arange(no_all_bins))[:, None]

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
    """This has to work"""
    labels_with_silence = num_class + 1

    # set the mask for the silence labels
    mask = labels == silence_label
    modified_labels = labels.clone()

    # set silence labels to max value
    modified_labels[mask] = num_class

    # one-hot encode the modified labels
    one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=labels_with_silence)

    return one_hot_labels


def mono_pitch_loss(logits, pitch, register_silence=False):
    # number of all bins, in / ex cluding silence
    no_all_bins = ppn.PITCH_BINS + int(register_silence)

    # transform logits of shape [B, O, T] into [B, T, O]
    logits = logits.permute(0, 2, 1).reshape(-1, no_all_bins)

    # start with a simple pitch_bins vector, make it one-hot
    pitch_bins = ppn.tools.frequency_to_bins(
            pitch, 
            register_silence=register_silence)

    # replace not finite (Nan or -inf, +inf) with a random value in [0, ppn.PITCH_BINS]
    pitch_bins_inf = ~pitch_bins.isfinite()
    no_silence = pitch_bins_inf.sum()
    pitch_bins[pitch_bins_inf] = torch.randint(0, ppn.PITCH_BINS, size = (no_silence,))

    pitch_bins_1hot = torch.nn.functional.one_hot(
            pitch_bins, 
            num_classes=no_all_bins)

    pitch_bins_1hot = pitch_bins_1hot.float()
    pitch_bins_1hot = pitch_bins_1hot.reshape(-1, no_all_bins)
    
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

    # return poly_pitch_loss(logits, pitch, model.register_silence)

    #if 'FretNetCrepe' in model.model_name():
    #    return poly_pitch_loss(logits, pitch, model.register_silence)
    #
    if 'MonoPitchNet1D' in model.model_name():
       return mono_pitch_loss(logits, pitch, model.register_silence)
