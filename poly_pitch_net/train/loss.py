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


def mono_pitch_loss(logits, pitch):
    # reshape [B, O, T] ==> [B, T, O]
    pass


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
    
    if 'mono' in model.model_name():
        return mono_pitch_loss(logits, pitch)
