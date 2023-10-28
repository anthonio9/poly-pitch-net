import torch

import poly_pitch_net as ppn


def loss(logits, bins, pitch_names):
    """
    Compute the CREPE loss for 6 channels

    Parameters
    ----------
    logits - output from the FretNetCrepe.forward() function.
    Should be of shape [B, C, O, T]

    bins - ground truth, values of the true pitch in semitones.

    look here for more 
    https://github.com/interactiveaudiolab/penn/blob/955656618bb71e1e2f040d1d134b8de834f51733/penn/train/core.py
    """
    # breakpoint()
    # reshape [B, C, O, T] ==> [B, C, T, O]
    logits = logits.reshape(shape=(
        logits.shape[0],
        logits.shape[1],
        logits.shape[-1],
        logits.shape[2]
        ))
    logits = logits.reshape(-1, pitch_names.shape[-1]).float()
    bins = bins.flatten()

    loss.cents = ppn.tools.convert.bins_to_cents(
            torch.arange(pitch_names.shape[-1]))[:, None]

    # Ensure values are on correct device (no-op if devices are the same)
    loss.cents = loss.cents.to(bins.device)

    # Create normal distributions
    distributions = torch.distributions.Normal(
            ppn.tools.convert.bins_to_cents(bins), 
            25)

    # Sample normal distributions
    bins = torch.exp(distributions.log_prob(loss.cents)).permute(1, 0)

    # Normalize
    bins = bins / (bins.max(dim=1, keepdims=True).values + 1e-8)
    
    # Compute binary cross-entropy loss
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        bins)
