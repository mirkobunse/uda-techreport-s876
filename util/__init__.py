import numpy as np
import pandas as pd
import sigma
import torch
from scipy.special import softmax
from tqdm.auto import tqdm

def lima_checkpoints(checkpoint_paths, data_dir, is_test_run=False):
    """Calculate the Li&Ma score for each checkpoint in a reason->path dictionary."""
    models = { p: torch.load(p) for p in np.unique(list(checkpoint_paths.values())) }
    y_pred = { m: np.empty(0) for m in models }
    metadata = pd.DataFrame()
    for X_night, metadata_night in data.readCrabDataYield(data_dir):
        metadata = pd.concat((metadata, metadata_night), ignore_index=True)
        if is_test_run:
            X_night = X_night[:12800] # only test the first 100 batches
            metadata = metadata[:12800]
        for m in models:
            y_night = predict_with_model(models[m], X_night)
            y_pred[m] = np.concatenate((y_pred[m], y_night))
        if is_test_run:
            break # only predict the first file during testing
    lima = { m: calculate_lima(y_pred[m], metadata) for m in models }
    return { reason: lima[checkpoint_paths[reason]] for reason in checkpoint_paths }

def calculate_lima(y_pred, metadata):
    """Calculate the Li&Ma score for a given set of predictions."""
    lima_df = metadata.copy().rename(columns={"event": "event_num"})[["run_id", "event_num", "night"]]
    lima_df["gamma_prediction"] = y_pred
    score, _ = sigma.lima(lima_df, prediction_threshold="optimize")
    return score

def predict_with_model(model, X):
    """Predict the data in batches, to not exceed the GPU memory."""
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    y_pred = np.empty(0)
    progressbar = tqdm(dataloader, total=int(len(dataloader)), ncols=80)
    model.eval()
    for (batch,) in progressbar:
        batch_output = model(batch.cuda(), 1)[0].detach().cpu()
        y_batch = softmax(batch_output, axis=1)[:, 0]
        y_pred = np.concatenate((y_pred, y_batch))
        progressbar.set_postfix(avg_y_pred=y_pred.mean())
    return y_pred
