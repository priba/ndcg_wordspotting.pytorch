# -*- coding: utf-8 -*-

"""
    Parse input arguments
"""

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate Word Spotting Rankings')

    # Dataset
    parser.add_argument('--dataset', default='gw')
    parser.add_argument('--partition', default='cv1', choices=['cv1', 'cv2', 'cv3', 'cv4'])
    parser.add_argument('--data_path', type=str)

    # Training
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', '-bz', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)

    # Loss
    parser.add_argument('--loss', default='ndcg', type=str, choices=('ndcg', 'map', 'combine'),
                        help="Select loss")
    parser.add_argument('--tau', default=1e-3, type=float, help='Smooth factor')
    parser.add_argument('--penalize', action='store_true', help='Penalize the relevance score')

    # Checkpoint
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    parser.add_argument('--early_stop', '-es', type=int, default=50, help='Early stopping epochs.')

    # Others
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--out_dim', default=64, type=int)

    return parser.parse_args()

