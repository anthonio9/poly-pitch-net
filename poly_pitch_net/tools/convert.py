import poly_pitch_net as ppn


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    # return penn.CENTS_PER_BIN * bins
    return ppn.CENTS_PER_BIN * bins
