import argparse
import numpy as np
import os
import pandas as pd
import torch
from model.ganin import GaninVGGNet, train_GaninVGGNet
from sklearn.model_selection import train_test_split
from util import data, lima_checkpoints

def main(
        output_path,
        checkpoint_dir,
        seed = 1,
        hidden_size_dc = 128,
        num_layers_dc = 2,
        gamma = 10.0,
        dc_weight = 0.2,
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

    # read real and simulated data
    Z_trn, Z_val = data.readCrabData(data_dir, seed)
    X_trn, y_trn = data.readSimData(data_dir)
    X_val, y_val = data.readSimTestData(data_dir)

    if is_test_run:
        Z_trn = Z_trn[:6400] # only test the first 100 batches
        Z_val = Z_val[:6400]
        X_trn = X_trn[:6400]
        X_val = X_val[:6400]
        y_trn = y_trn[:6400]
        y_val = y_val[:6400]

    # configure the model
    model = GaninVGGNet(hidden_size_dc=hidden_size_dc, num_layers_dc=num_layers_dc)
    model.cuda()

    # train a model to collect the paths of all checkpoints
    checkpoint_paths = train_GaninVGGNet(
        model,
        checkpoint_dir,
        X_trn,
        y_trn,
        X_val,
        y_val,
        Z_trn,
        Z_val,
        gamma = gamma,
        dc_weight = dc_weight,
        n_epochs = n_epochs
    )

    # evaluate all checkpoints
    lima = lima_checkpoints(checkpoint_paths, data_dir, is_test_run)

    # store the results
    df = pd.DataFrame({ 'checkpoint': lima.keys(), 'lima': lima.values() })
    df['hidden_size_dc'] = hidden_size_dc
    df['num_layers_dc'] = num_layers_dc
    df['gamma'] = gamma
    df['dc_weight'] = dc_weight
    df['seed'] = seed
    df.to_csv(output_path)
    print(f"LiMa scores succesfully stored at {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('checkpoint_dir', type=str, help='directory for model checkpoints')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random number generator seed (default: 1)')
    parser.add_argument('--hidden_size_dc', type=int, default=128, metavar='N',
                        help='size of domain classification layers (default: 128)')
    parser.add_argument('--num_layers_dc', type=int, default=2, metavar='N',
                        help='number of domain classification layers (default: 2)')
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='gamma parameter of the gradient reversal layer (default: 10.0)')
    parser.add_argument('--dc_weight', type=float, default=0.2,
                        help='weight of the domain classification loss (default: 0.2)')
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
        hidden_size_dc = args.hidden_size_dc,
        num_layers_dc = args.num_layers_dc,
        gamma = args.gamma,
        dc_weight = args.dc_weight,
        n_epochs = args.n_epochs,
        data_dir = args.data_dir,
        is_test_run = args.is_test_run
    )
