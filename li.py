import argparse
import numpy as np
import os
import pandas as pd
import shutil
import torch
from model.buschjaeger import BaselineVGGNet
from util import data, lima_checkpoints
from tqdm.auto import tqdm

def train_top_layers(checkpoint_model, max_k):
    checkpoint_model.eval()
    if max_k <= 0:
        max_k = len(checkpoint_model.layers)
    k = 0 # index of the batch norm layer, counted from the top
    for layer in reversed(checkpoint_model.layers):
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            if k < max_k:
                layer.train()
                k += 1
                if k == max_k:
                    break
    print(f"Re-training the top {k} batch norm layers")
    return k

def main(
        output_path,
        checkpoint_dir,
        checkpoint,
        seed = 1,
        n_epochs = 1,
        max_k = 0,
        data_dir = '/mnt/data/data',
        is_test_run = False
    ):
    print(f"Starting an experiment to produce {output_path} from {checkpoint}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ensure that the checkpoint directory is empty and exists
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # only real data is needed for the adaptation
    Z_all = np.concatenate(data.readCrabData(data_dir, seed)) # Z_trn + Z_val
    X_val, y_val = data.readSimTestData(data_dir) # validate

    if is_test_run:
        Z_all = Z_all[:6400] # only test the first 100 batches
        X_val = X_val[:6400]
        y_val = y_val[:6400]

    # dataloader for adaptation (target) data
    ds_Z = torch.utils.data.TensorDataset(
        torch.tensor(Z_all, dtype=torch.float32).unsqueeze(1)
    )
    dl_Z = torch.utils.data.DataLoader(ds_Z, batch_size=128, shuffle=True)

    # dataloader for validation (source) data
    ds_Xy_val = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val)
    )
    dl_Xy_val = torch.utils.data.DataLoader(ds_Xy_val, batch_size=128, shuffle=True)

    # adapt each checkpoint
    model = BaselineVGGNet() # torch.load requires this variable
    checkpoint_model = torch.load(checkpoint).cuda()
    criterion = torch.nn.CrossEntropyLoss(reduction="none") # only for validation

    k = max_k
    best_val_loss = np.inf
    val_loss_history = []
    val_acc_history = []
    history_path = f"{checkpoint_dir}/validation_loss_history.csv"
    for epoch in range(n_epochs):
        # adapt the batch normalization layers
        k = train_top_layers(checkpoint_model, max_k) # set k top BN layers to .train()
        progressbar = tqdm(
            dl_Z,
            desc = f"Adaptation {epoch+1}/{n_epochs}",
            total = int(len(dl_Z)),
            ncols = 80
        )
        for (Z_i,) in progressbar:
            checkpoint_model(Z_i.cuda(), 1) # update the .train() BN layers

        # validate
        checkpoint_model.eval() # update none of the layers this time
        running_loss = 0.0
        running_acc = 0.0
        running_count = 0
        progressbar = tqdm(
            dl_Xy_val,
            desc = "Validation",
            total = int(len(dl_Xy_val)),
            ncols = 80
        )
        for (X_i, y_i) in progressbar:
            y_pred, _ = checkpoint_model(X_i.cuda())
            running_loss += criterion(y_pred, y_i.cuda()).sum().item()
            y_pred = y_pred.argmax(1).detach().cpu()
            running_acc += np.sum(y_pred.numpy() == y_i.numpy())
            running_count += len(X_i)
        val_loss = running_loss / running_count
        val_loss_history.append(val_loss)
        val_acc = running_acc / running_count
        val_acc_history.append(val_acc)
        print(f"Epoch {epoch+1}: val loss {val_loss} val acc {val_acc}")

        # collect reasons for creating a checkpoint from this epoch
        reasons_to_save = []
        if (epoch+1) == n_epochs: # save the final epoch
            reasons_to_save.append(f"finalEpoch")
        if val_loss < best_val_loss:
            reasons_to_save.append(f"bestValLoss")
            best_val_loss = val_loss

        # store the model only once
        if len(reasons_to_save) > 0:
            checkpoint_path = f"{checkpoint_dir}/epoch{epoch+1:03d}.pth"
            print(f"Saving model at {checkpoint_path} {reasons_to_save}")
            torch.save(checkpoint_model, checkpoint_path)
            for reason in reasons_to_save:
                reason_path = f"{checkpoint_dir}/{reason}.pth"
                if os.path.exists(reason_path):
                    os.remove(reason_path) # os.symlink errors if the path exists
                os.symlink(checkpoint_path, reason_path)

        # store the loss history
        print(f"Updating {history_path}")
        pd.DataFrame({
            "epoch": np.arange(epoch + 1) + 1,
            "val_loss": val_loss_history,
            "val_acc": val_acc_history,
        }).to_csv(history_path)

    # evaluate all checkpoints
    print("Training complete!")
    checkpoint_paths = {
        reason: os.readlink(f"{checkpoint_dir}/{reason}.pth") for reason in [
            "finalEpoch",
            "bestValLoss",
        ]
    } # follow symlinks of all reasons_to_save
    lima = lima_checkpoints(checkpoint_paths, data_dir, is_test_run)

    # store the results
    df = pd.DataFrame({ 'checkpoint': lima.keys(), 'lima': lima.values() })
    df['seed'] = seed
    df['k'] = k
    df['input_checkpoint'] = checkpoint
    df.to_csv(output_path)
    print(f"LiMa scores succesfully stored at {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('checkpoint_dir', type=str, help='directory for model checkpoints')
    parser.add_argument('checkpoint', type=str, help='checkpoint to adapt')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random number generator seed (default: 1)')
    parser.add_argument('--n_epochs', type=int, default=1, metavar='N',
                        help='number of adaptation epochs (default: 1)')
    parser.add_argument('--max_k', type=int, default=0, metavar='N',
                        help='maximum number of top BN layers to adapt (default: 0 = all)')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/data',
                        help='directory of *.npy.gz files (default: /mnt/data/data)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.checkpoint_dir,
        args.checkpoint,
        seed = args.seed,
        n_epochs = args.n_epochs,
        max_k = args.max_k,
        data_dir = args.data_dir,
        is_test_run = args.is_test_run
    )
