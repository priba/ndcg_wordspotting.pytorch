# -*- coding: utf-8 -*-

"""
    Parse input arguments
"""

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate Word Spotting Rankings')

    # Dataset
    parser.add_argument('--dataset', default='gw')
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

    # Others
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--out_dim', default=64, type=int)

    return parser.parse_args()

