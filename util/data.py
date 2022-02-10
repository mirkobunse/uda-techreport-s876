import numpy as np
import os
import pandas as pd
from glob import glob
from gzip import GzipFile
from sklearn.model_selection import train_test_split

def readCrabData(dir, seed=None):
    """Read a cached training test split or all Crab samples."""
    if seed == None:
        return __readCrabData(dir)
    try:
        Z_trn = np.load(f"{dir}/.cache/trn_{seed}.npy", allow_pickle=True)
        Z_val = np.load(f"{dir}/.cache/val_{seed}.npy", allow_pickle=True)
    except:
        print(f"ERROR: cannot load from {dir}/.cache; solve the issue with 'make cache'")
        raise # rethrow the current exception as it is
    return Z_trn, Z_val

def storeCrabSplits(dir, seeds):
    """Store training test splits for readCrabData(dir, seed)."""
    os.makedirs(f"{dir}/.cache", exist_ok=True)
    Z = __readCrabData(dir)
    for seed in seeds:
        np.random.seed(seed)
        Z_seed = Z[np.random.permutation(len(Z))[:300000]] # same size as simulations
        Z_trn, Z_val = train_test_split(Z_seed, test_size=1/3)
        print(f"Saving {dir}/.cache/{{trn,val}}_{seed}.npy")
        np.save(f"{dir}/.cache/trn_{seed}.npy", Z_trn, allow_pickle=True)
        np.save(f"{dir}/.cache/val_{seed}.npy", Z_val, allow_pickle=True)

def __readCrabData(dir):
    """Read the observed crab samples to be predicted."""
    crab = np.empty((0,46,45))
    files_Z = sorted(glob(f"{dir}/factCrabExamples_no_cut_*.npy.gz")) # files with crab samples
    for file_Z in files_Z:
        print("Reading", file_Z)
        Z = np.load(GzipFile(file_Z,"rb"))
        print(Z.shape)
        crab = np.append(crab, Z, axis=0)
    return crab

def readSimData(dir, with_cut=False):
    """Read the training data (as in readData.ipynb)."""
    cut = "with_cut" if with_cut else "no_cut"
    X = np.load(f"{dir}/factTrainExamples_{cut}.npy", allow_pickle=True)
    y = np.load(f"{dir}/factTrainTargets_{cut}.npy", allow_pickle=True)
    return X, y

def readSimTestData(dir, with_cut=False):
    """Read the test data (as in readData.ipynb)."""
    cut = "with_cut" if with_cut else "no_cut"
    X = np.load(f"{dir}/factTestExamples_{cut}.npy", allow_pickle=True)
    y = np.load(f"{dir}/factTestTargets_{cut}.npy", allow_pickle=True)
    return X, y

def readCrabDataYield(dir):
    """Generator for the observed crab samples to be predicted."""
    files_Z = sorted(glob(f"{dir}/factCrabExamples_no_cut_*.npy.gz")) # files with crab samples
    files_m = sorted(glob(f"{dir}/factCrabMeta_no_cut_*.npy.gz")) # files with crab meta-data
    for file_Z, file_m in zip(files_Z, files_m):
        print("Reading", file_Z, "and", file_m)
        Z = np.load(GzipFile(file_Z,"rb"))
        m = pd.DataFrame(np.load(GzipFile(file_m,"rb"), allow_pickle=True).tolist())
        yield Z, m # return a tuple of samples (46x45 pixels) and their meta-data (DataFrames with 6 columns)
