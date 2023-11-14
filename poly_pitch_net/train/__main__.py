import argparse
import poly_pitch_net as ppn


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['poly', 'mono1d'],
        help='Type of the trained model, polyphonic or monophonic',
        required=True)
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')
    parser.add_argument(
        '-s', 
        '--silence',
        action=register_silence,
        help="Enable silence registration")

    return parser.parse_known_args()[0]


ppn.train.run(**vars(parse_args()))
