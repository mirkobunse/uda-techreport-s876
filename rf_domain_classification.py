import argparse
import numpy as np
import os
import pandas as pd
from sigma import skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.utils import resample, shuffle

def main(
        n_samples = 200000,
        seed = 1,
        data_dir = '/mnt/data/data'
    ):
    print(f"RNG seed: {seed}")
    np.random.seed(seed)

    # read simulated and observed data
    X_sim, y_sim = skl.get_training_data()
    X_obs = skl.get_crab_data()
    print(f"Sampling {n_samples} out of {len(X_sim)} simulated and {len(X_obs)} observed samples")
    X, y = shuffle(
        np.concatenate([
            resample(X_sim[y_sim==0], replace=False, n_samples=int(n_samples/4)),
            resample(X_sim[y_sim==1], replace=False, n_samples=int(n_samples/4)),
            resample(X_obs, replace=False, n_samples=int(n_samples/2))
        ]), # draw n_samples 50/50 from both data sources
        np.concatenate([
            np.zeros(int(n_samples/2), dtype=int),
            np.ones(int(n_samples/2), dtype=int)
        ]) # construct domain labels y
    ) # ...and shuffle (X, y)

    # https://github.com/fact-project/open_crab_sample_analysis/blob/f40c4fab57a90ee589ec98f5fe3fdf38e93958bf/configs/aict.yaml#L66
    clf = RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1, max_depth=15)
    print("Performing a 10-fold CV with a 200-member random forest")
    scores = cross_validate(clf, X, y, cv=10)
    print(f"Avg domain classification accuracy: {scores['test_score'].mean()}")
    print(f"Scores: {scores['test_score']}")
    print(f"Fit time: {scores['fit_time']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200000, metavar='N',
                        help='number of samples to consider (default: 10000)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random number generator seed (default: 1)')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/data',
                        help='directory of data files (default: /mnt/data/data)')
    args = parser.parse_args()
    main(args.n_samples, args.seed, args.data_dir)
