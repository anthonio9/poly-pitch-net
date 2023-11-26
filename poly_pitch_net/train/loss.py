import poly_pitch_net as ppn
import torch



def poly_pitch_loss(logits, pitch, register_silence=False, loss_type: str=ppn.LOSS_ONE_HOT):
    # number of all bins, in / ex cluding silence
    no_all_bins = ppn.PITCH_BINS + int(register_silence)

    # transform logits of shape [B, O, T] into [B, T, O]
    logits = logits.permute(0, 2, 1).reshape(-1, no_all_bins)

    # start with a simple pitch_bins vector, make it one-hot
    pitch_bins = ppn.tools.frequency_to_bins(
            pitch, 
            register_silence=register_silence).long()

    # replace zero pitch with a random value in [0, ppn.PITCH_BINS]
    no_silence = (pitch == 0).sum()
    random_bins = torch.randint(0, ppn.PITCH_BINS, size = (no_silence,))
    pitch_bins[pitch == 0] = random_bins.to(logits.device)

    if loss_type == ppn.LOSS_ONE_HOT:
        pitch_bins_1hot = torch.nn.functional.one_hot(
                pitch_bins, 
                num_classes=no_all_bins)

        pitch_bins_1hot = pitch_bins_1hot.float()
        pitch_bins_1hot = pitch_bins_1hot.reshape(-1, no_all_bins)
        pitch_bins = pitch_bins_1hot

    elif loss_type == ppn.LOSS_GAUSS:
        pitch_bins = ppn.tools.bins_to_cents(pitch_bins)

        loss.cents = ppn.tools.convert.bins_to_cents(
                torch.arange(no_all_bins))[:, None]

        # Ensure values are on correct device (no-op if devices are the same)
        loss.cents = loss.cents.to(pitch.device)

        pitch_bins = pitch_bins.flatten()

        # Create normal distributions
        distributions = torch.distributions.Normal(
                pitch_bins, 
                25)

        # Sample normal distributions
        pitch_bins = torch.exp(distributions.log_prob(loss.cents)).permute(1, 0)

        # Normalize
        pitch_bins = pitch_bins / (pitch_bins.max(dim=1, keepdims=True).values + 1e-8)
    
    # Compute binary cross-entropy loss
    return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            pitch_bins)

def mono_pitch_loss(logits, pitch, register_silence=False):
    # number of all bins, in / ex cluding silence
    no_all_bins = ppn.PITCH_BINS + int(register_silence)

    # transform logits of shape [B, O, T] into [B, T, O]
    logits = logits.permute(0, 2, 1).reshape(-1, no_all_bins)

    # start with a simple pitch_bins vector, make it one-hot
    pitch_bins = ppn.tools.frequency_to_bins(
            pitch, 
            register_silence=register_silence).long()

    # replace zero pitch with a random value in [0, ppn.PITCH_BINS]
    no_silence = (pitch == 0).sum()
    random_bins = torch.randint(0, ppn.PITCH_BINS, size = (no_silence,))
    pitch_bins[pitch == 0] = random_bins.to(logits.device)
    pitch_bins_1hot = torch.nn.functional.one_hot(
            pitch_bins, 
            num_classes=no_all_bins)

    pitch_bins_1hot = pitch_bins_1hot.float()
    pitch_bins_1hot = pitch_bins_1hot.reshape(-1, no_all_bins)
    
    # Compute binary cross-entropy loss
    #return torch.nn.functional.binary_cross_entropy_with_logits(
    #        logits,
    #        pitch_bins_1hot)

    return torch.nn.functional.cross_entropy(
            logits,
            pitch_bins_1hot)


def loss(model, logits, pitch, pitch_names=None, loss_type: str=ppn.LOSS_ONE_HOT):
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

    return poly_pitch_loss(logits, pitch, model.register_silence,
            loss_type=loss_type)

    #if 'FretNetCrepe' in model.model_name():
    #    return poly_pitch_loss(logits, pitch, model.register_silence)
    #
    #if 'MonoPitchNet1D' in model.model_name():
    #   return mono_pitch_loss(logits, pitch, model.register_silence)
