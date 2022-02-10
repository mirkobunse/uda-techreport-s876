import argparse
import numpy as np
import os
import pandas as pd
import torch
from model.buschjaeger import BaselineVGGNet, train_BaselineVGGNet
from sklearn.model_selection import train_test_split
from util import data, lima_checkpoints

def main(
        output_path,
        checkpoint_dir,
        seed = 1,
        n_epochs = 25,
        data_dir = '/mnt/data/data',
        is_test_run = False
    ):
    print(f"Starting an experiment to produce {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # read simulated data
    X_trn, y_trn = data.readSimData(data_dir)
    X_val, y_val = data.readSimTestData(data_dir)

    if is_test_run:
        X_trn = X_trn[:6400]
        y_trn = y_trn[:6400]
        X_val = X_val[:6400]
        y_val = y_val[:6400]

    # configure the model
    model = BaselineVGGNet()
    model.cuda()

    # train a model to collect the paths of all checkpoints
    checkpoint_paths = train_BaselineVGGNet(
        model,
        checkpoint_dir,
        X_trn,
        y_trn,
        X_val,
        y_val,
        n_epochs = n_epochs
    )

    # evaluate all checkpoints
    lima = lima_checkpoints(checkpoint_paths, data_dir, is_test_run)

    # store the results
    df = pd.DataFrame({ 'checkpoint': lima.keys(), 'lima': lima.values() })
    df['seed'] = seed
    df.to_csv(output_path)
    print(f"LiMa scores succesfully stored at {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('checkpoint_dir', type=str, help='directory for model checkpoints')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random number generator seed (default: 1)')
    parser.add_argument('--n_epochs', type=int, default=25, metavar='N',
                        help='number of training epochs (default: 25)')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/data',
                        help='directory of *.npy.gz files (default: /mnt/data/data)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.checkpoint_dir,
        seed = args.seed,
        n_epochs = args.n_epochs,
        data_dir = args.data_dir,
        is_test_run = args.is_test_run
    )
