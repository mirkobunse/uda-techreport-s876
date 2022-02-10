# Unsupervised Domain Adaptation for Gamma Hadron Separation

This repository contains the experiments for our technical report

```bibtex
@TechReport{drew2022unsupervised,
  title       = {Unsupervised Domain Adaptation for Gamma Hadron Separation},
  author      = {Robin Drew and Mirko Bunse},
  institution = {TU Dortmund University},
  year        = {2022},
  number      = {1}
}
```

where we evaluate deep unsupervised domain adaptation (UDA) techniques in the context of gamma hadron separation, an important prediction task for imaging air Cherenkov telescopes. Namely, we consider UDA components are proposed by [Ganin et al. (2015)](http://proceedings.mlr.press/v37/ganin15.html) and by [Li et al. (2018)](https://openreview.net/pdf?id=Hk6dkJQFx) as an extension to the non-UDA baseline proposed by [Buschjäger et al. (2020)](https://link.springer.com/content/pdf/10.1007%2F978-3-030-67667-4_29.pdf).

**Feel free to file an issue for any question you have!**


## Experiments

**Caution:** the following description is only valid for a Docker container. See `docker/README.md` for a description of the Docker setup.

You can start all experiments of a single trial by calling `make` with the respective target. Each trial is identified by a random number generator seed (between 0 and 9):

```
make seed0
...
make seed9
```

We recommend you inspect all steps taken by `make` with the `-n` (= `--dry-run`) switch, before actually starting them:

```
make -n
```

**Pre-computing training test splits:** if the first step of `make` is `python generate_cache.py`, you might want to execute this step separately before all others, particularly if you intend to execute multiple seeds in parallel. This first step generates subsamples of the Crab nebula data in the `/mnt/data/data/.cache/` directory, which provide a tremendous speed-up for the initialization of the experiments. There would be storage collisions if this step was called multiple times in parallel.

**Checkpoints and significance scores:** the trained models from each experiment are stored as `/mnt/data/checkpoints/*/*.pth` and the significance scores are stored as `/mnt/data/results/*.csv`.

**Mount points:** The `/mnt/data/` directory is a mount point inside a Docker container, where the `docker/run.sh` script mounts the NFS directory `s876home:/rdata/s01f_c3_004/uda`. This directory is specific to our setup at the SFB 876; you might need to change this setting in the `docker/run.sh` script.

## Plots and Tables

Generating the plots and tables from our report require all experiments (`make seed0`, ..., `make seed9`) to be completed. You can then call the aggregation scripts which produce the plots and tables automatically:

```
python generate_table.py table_01.tex
python generate_roc.py

```

You find a documentation of all command line arguments by adding the `-h` switch to the calls above. Here, `generate_roc.py` only produces the CSVs from which a plot can be generated; we have used `pgfplots` for this purpose.
