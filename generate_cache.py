import argparse
import numpy as np
from util import data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10, metavar='N',
                        help='number of seeds with which to split (default: 10)')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/data',
                        help='directory of *.npy.gz files (default: /mnt/data/data)')
    args = parser.parse_args()

    print(f"Generating the splits of {args.n_seeds} seeds in {args.data_dir}/.cache/")
    data.storeCrabSplits(args.data_dir, np.arange(args.n_seeds))
