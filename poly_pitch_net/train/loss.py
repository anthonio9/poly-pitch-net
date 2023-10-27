import torch

import poly_pitch_net as ppn


def loss(logits, bins):
    """
    Compute the CREPE loss for 6 channels

    Parameters
    ----------
    logits - output from the FretNetCrepe.post_proc() function.
    Should be of shape [B, T, C, 1]

    """

    loss.cents = ppn.tools.convert.bins_to_cents(
                torch.arange(penn.PITCH_BINS))[:, None] 

    # Ensure values are on correct device (no-op if devices are the same)
    loss.cents = loss.cents.to(bins.device)

    # look here for more 
    # https://github.com/interactiveaudiolab/penn/blob/955656618bb71e1e2f040d1d134b8de834f51733/penn/train/core.py#L159 loss()


    # Compute binary cross-entropy loss
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        bins)
