import argparse
import numpy as np
import pandas as pd
from functools import partial

CONFIGURATIONS = [ # default configurations
    "buschjaeger",
    "ganin_g10_w02",
    "ganin_g20_w01",
    "li_k0",
    "li_k1"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path to store the ROC plot CSV')
    parser.add_argument('--n_seeds', type=int, default=10, metavar='N',
                        help='number of seeds from which to collect results (default: 10)')
    parser.add_argument('--seeds', type=int, default=[], metavar='N', nargs='*',
                        help='set of seeds to consider (overwrites n_seeds)')
    parser.add_argument('--configurations', default=CONFIGURATIONS, nargs='*',
                        help=f'set of configurations to consider (default: {CONFIGURATIONS})')
    parser.add_argument('--results_dir', type=str, default='/mnt/data/results',
                        help='directory of *.npy.gz files (default: /mnt/data/results)')
    args = parser.parse_args()
    seeds = np.arange(args.n_seeds) if len(args.seeds) == 0 else np.array(args.seeds)
    seeds_str = 's{' + ','.join(seeds.astype(str)) + '}'

    # read the results of a single trial
    def read_results(configuration, seed):
        df = pd.read_csv(f"{args.results_dir}/{configuration}_s{seed}.csv")
        df["seed"] = seed
        df["configuration"] = configuration
        return df

    # read all results of a configuration
    def read_all_results(configuration):
        print(f"Reading {args.results_dir}/{configuration}_{seeds_str}.csv")
        f = partial(read_results, configuration)
        return pd.concat(map(f, seeds))

    # read and aggregate
    results = []
    for c in args.configurations:
        results.append(read_all_results(c))
    df = pd.concat(results)
    df_b = df[(df["configuration"]=="buschjaeger")*(df["checkpoint"]=="bestValLoss")]
    df["wins"] = [r["lima"] > df_b[df_b["seed"]==r["seed"]]["lima"].to_numpy()[0] for _, r in df.iterrows()]
    print(df[df["seed"]==0])
    print(df[df["seed"]==1])
    agg = df.groupby(["configuration", "checkpoint"]).agg(
        lima = ("lima", "mean"),
        lima_std = ("lima", "std"),
        lima_max = ("lima", "max"),
        lima_min = ("lima", "min"),
        wins = ("wins", "sum")
    )
    print(agg)

    # LaTeX format
    agg.reset_index(inplace=True)
    print(f"Storing aggregated results at {args.output_path}")
    agg["configuration"].replace(inplace=True, to_replace={
        "buschjaeger": "Buschj\\\"ager et al.",
        "ganin_g10_w02": "Ganin et al. ($\\gamma=10, w=0.2$)",
        "ganin_g20_w01": "Ganin et al. ($\\gamma=20, w=0.1$)",
        "li_k0": "Li et al. (all BN layers)",
        "li_k1": "Li et al. (top BN layer)"
    }) # rename configurations
    agg["checkpoint"].replace(inplace=True, to_replace={
        "bestValLoss": "best loss",
        "bestLabelLoss": "best label loss",
        "bestOverallLoss": "best overall loss",
        "finalEpoch": "final epoch"
    }) # rename checkpoints
    best_avg = np.where(agg["lima"] == agg["lima"].max())[0]
    best_max = np.where(agg["lima_max"] == agg["lima_max"].max())[0]
    best_min = np.where(agg["lima_min"] == agg["lima_min"].max())[0]
    best_wins = np.where(agg["wins"] == agg["wins"].max())[0]
    with open(args.output_path, "w") as f:
        f.write("\\begin{tabular}{rlcccc}\n")
        f.write("  \\toprule\n") # \usepackage{booktabs}
        f.write(" & ".join([
            "  method", # with indentation
            "checkpoint",
            "avg. $s(n_\\text{on}, n_\\text{off})$",
            "max.",
            "min.",
            "wins \\\\\n"
        ]))
        f.write("  \\midrule\n")
        last_configuration = None
        for i_row, row in agg.iterrows():
            if row["configuration"] != last_configuration:
                f.write("  " + row["configuration"] + "\n")
                last_configuration = row["configuration"]
            f.write("    & " + row["checkpoint"] + " & $")
            if i_row in best_avg:
                f.write("\\mathbf{")
            f.write(f"{row['lima']:.3f} \\pm {row['lima_std']:.3f}")
            if i_row in best_avg:
                f.write("}")
            f.write("$ & $")
            if i_row in best_max:
                f.write("\\mathbf{")
            f.write(f"{row['lima_max']:.3f}")
            if i_row in best_max:
                f.write("}")
            f.write("$ & $")
            if i_row in best_min:
                f.write("\\mathbf{")
            f.write(f"{row['lima_min']:.3f}")
            if i_row in best_min:
                f.write("}")
            if row["configuration"]=="Buschj\\\"ager et al." and row["checkpoint"]=="best loss":
                f.write("$ & --- \\\\\n")
            else:
                f.write("$ & $")
                if i_row in best_wins:
                    f.write("\\mathbf{")
                f.write(str(row['wins']))
                if i_row in best_wins:
                    f.write("}")
                f.write("$ \\\\\n")
        f.write("  \\bottomrule\n")
        f.write("\\end{tabular}\n")
