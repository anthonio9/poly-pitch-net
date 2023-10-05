def get_output_size(in_size=9, ks=3, dl=3, pad=1, stride=1):
    fks = ks + (ks - 1)*(dl - 1)
    out_size = 1 + (in_size - fks + pad*2)/stride

    return out_size
