import argparse
import numpy as np
import os
import pandas as pd
import torch
from model.buschjaeger import BaselineVGGNet
from model.ganin import GaninVGGNet
from util import data
from scipy.special import softmax
from sklearn.metrics import roc_curve
from tqdm.auto import tqdm

CONFIGURATIONS = [ # default configurations
    "buschjaeger",
    "ganin_g10_w02",
    "ganin_g20_w01",
    "li_k0",
    "li_k1"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/mnt/data/roc',
                        help='directory to store the ROC plot CSVs')
    parser.add_argument('--seeds', type=int, default=[5], metavar='N', nargs='*',
                        help='set of seeds to consider (default: [5])')
    parser.add_argument('--n_thresholds', type=int, default=1000, metavar='N',
                        help='number of thresholds to plot (default: 1000)')
    parser.add_argument('--configurations', default=CONFIGURATIONS, nargs='+',
                        help=f'configurations to consider (default: {CONFIGURATIONS})')
    parser.add_argument('--checkpoint_dir', type=str, default='/mnt/data/checkpoints',
                        help='base directory for all model checkpoints')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/data',
                        help='directory of *.npy.gz files (default: /mnt/data/data)')
    args = parser.parse_args()
    seeds = np.array(args.seeds)
    seeds_str = 's{' + ','.join(seeds.astype(str)) + '}'

    # ensure that the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # read the validation data
    X_val, y_val = data.readSimTestData(args.data_dir)
    ds_X_val = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    )
    dl_X_val = torch.utils.data.DataLoader(ds_X_val, batch_size=128, shuffle=False)

    for configuration in args.configurations:
        model = GaninVGGNet() if configuration.startswith("ganin") else BaselineVGGNet()
        checkpoint = "bestLabelLoss" if configuration.startswith("ganin") else "bestValLoss"

        for seed in seeds:
            checkpoint_path = f"{args.checkpoint_dir}/{configuration}_s{seed}/{checkpoint}.pth"
            output_path = f"{args.output_dir}/{configuration}_s{seed}_{checkpoint}.csv"
            checkpoint_model = torch.load(checkpoint_path).cuda().eval()

            print(f"Generating an ROC plot in {output_path}")
            y_i = [] # list of batch-wise outputs
            with torch.no_grad():
                for (X_i,) in tqdm(dl_X_val, total=int(len(dl_X_val)), ncols=80):
                    y_i.append(checkpoint_model(X_i.cuda(), 1)[0].softmax(axis=1)[:,1].cpu())
            y_pred = np.concatenate(y_i)
            fpr, tpr, thresholds = roc_curve(y_val, y_pred)

            # reduce to n_thresholds
            i_thresholds = np.minimum(
                np.round(np.arange(args.n_thresholds) * (len(thresholds) / (args.n_thresholds-1))).astype(int),
                len(thresholds) - 1
            )
            pd.DataFrame({
                'fpr': fpr[i_thresholds],
                'tpr': tpr[i_thresholds],
                'thresholds': thresholds[i_thresholds]
            }).to_csv(output_path)
