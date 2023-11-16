import argparse
from pathlib import Path

import poly_pitch_net as ppn


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=Path,
        help='Path to the model to evaluate',
        required=True)
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')
    parser.add_argument(
        '-s', 
        '--register_silence',
        action="store_true",
        help="Enable silence registration")

    return parser.parse_known_args()[0]


ppn.evaluate.run_evaluation(**vars(parse_args()))
