import typing as tp
import argparse


def base_training_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputsize', type=int,   default=512)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--epochs',    type=int,   default=30)
    parser.add_argument(
        '--checkpointdir', type=str, default='./checkpoints', help='Where to save trained models'
    )
    return parser
