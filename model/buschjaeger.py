import numpy as np
import os
import pandas as pd
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Function
from tqdm.auto import tqdm

class Flatten(nn.Module):
    """Custom layer copied from https://github.com/sbuschjaeger/Pysembles"""
    def __init__(self, store_shape=False):
        super(Flatten, self).__init__()
        self.store_shape = store_shape
    def forward(self, x):
        if self.store_shape:
            self.shape = x.shape
        return x.flatten(1)

class BaselineVGGNet(nn.Module):
    """
    The baseline VGGNet [buschjaeger2020onsite], copied from
    https://github.com/sbuschjaeger/Pysembles
    """
    def __init__(
            self,
            input_size = (1, 46, 45),
            n_channels = 128,
            depth = 4,
            hidden_size = 512,
            p_dropout = 0.3,
            n_classes = 2
        ):
        super().__init__()
        in_channels = input_size[0]
        layers = []
        for level in range(depth):
            layers.extend([
                nn.Conv2d(
                    in_channels if level == 0 else level * n_channels,
                    (level+1) * n_channels,
                    kernel_size = 3,
                    padding = 1,
                    stride = 1,
                    bias = True
                ),
                nn.BatchNorm2d((level+1) * n_channels),
                nn.ReLU(),
                nn.Conv2d(
                    (level+1) * n_channels,
                    (level+1) * n_channels,
                    kernel_size = 3,
                    padding = 1,
                    stride = 1,
                    bias = True
                ),
                nn.BatchNorm2d((level+1) * n_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        # determine the size of the linear layer
        x = torch.rand((1, *input_size)).type(torch.FloatTensor)
        for l in layers:
            x = l(x)
        lin_size = x.view(1, -1).size()[1]

        # set up the label predictor
        layers.extend([
            Flatten(),
            nn.Linear(lin_size, hidden_size),
            nn.Dropout(p_dropout) if p_dropout > 0 else None,
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        ])
        layers = filter(None, layers)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, dummy_alpha=None):
        return self.layers(x), None


def train_BaselineVGGNet(
        model,
        checkpoint_dir,
        X_trn,
        y_trn,
        X_val,
        y_val,
        n_epochs = 25,
        lr = 1e-4
    ):
    """Train a BaselineVGGNet model."""
    print(f"BaselineVGGNet: starting the training for {checkpoint_dir}/*.pth")
    history_path = f"{checkpoint_dir}/validation_loss_history.csv"

    # ensure that the checkpoint directory is empty and exists
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # dataloader for training (source) data
    ds_Xy_trn = torch.utils.data.TensorDataset(
        torch.tensor(X_trn, dtype=torch.float32).unsqueeze(1), # add a dimension at axis 1
        torch.tensor(y_trn) # ..and cast both X and y as tensors
    )
    dl_Xy_trn = torch.utils.data.DataLoader(ds_Xy_trn, batch_size=128, shuffle=True)

    # dataloader for validation (source) data
    ds_Xy_val = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1), # add a dimension at axis 1
        torch.tensor(y_val) # ..and cast both X and y as tensors
    )
    dl_Xy_val = torch.utils.data.DataLoader(ds_Xy_val, batch_size=128, shuffle=True)

    # the loss function and the optimizer
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)



    best_val_loss = np.inf
    trn_loss_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(n_epochs): # loop over the entire dataset multiple times
        model.train()
        running_loss = 0.0
        running_count = 0
        progressbar = tqdm(
            dl_Xy_trn,
            desc = f"Epoch {epoch+1}/{n_epochs}",
            total = int(len(dl_Xy_trn)),
            ncols = 80
        )

        # loop over all batches
        for (X_i, y_i) in progressbar:
            optimizer.zero_grad() # reset the gradients
            y_pred, _ = model(X_i.cuda())
            loss = criterion(y_pred, y_i.cuda())
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.sum().item()
            running_count += len(X_i)
            progressbar.set_postfix(loss=running_loss/running_count)
        trn_loss = running_loss / running_count
        trn_loss_history.append(trn_loss)

        # validation set accuracy with a progress bar
        model.eval() # validate on the hold-out set
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
            y_pred, _ = model(X_i.cuda())
            running_loss += criterion(y_pred, y_i.cuda()).sum().item()
            y_pred = y_pred.argmax(1).detach().cpu()
            running_acc += np.sum(y_pred.numpy() == y_i.numpy())
            running_count += len(X_i)
        val_loss = running_loss / running_count
        val_loss_history.append(val_loss)
        val_acc = running_acc / running_count
        val_acc_history.append(val_acc)
        scheduler.step(val_loss)
        print(f"BaselineVGGNet [epoch {epoch+1}]: trn loss {trn_loss} val loss {val_loss} val acc {val_acc}")

        # collect reasons for creating a checkpoint from this epoch
        reasons_to_save = []
        if (epoch+1) == n_epochs: # save the final epoch
            reasons_to_save.append('finalEpoch')
        if val_loss < best_val_loss:
            reasons_to_save.append('bestValLoss')
            best_val_loss = val_loss

        # store the model only once
        if len(reasons_to_save) > 0:
            checkpoint_path = f"{checkpoint_dir}/epoch{epoch+1:03d}.pth"
            print(f"BaselineVGGNet: saving model at {checkpoint_path} {reasons_to_save}")
            torch.save(model, checkpoint_path)
            for reason in reasons_to_save:
                reason_path = f"{checkpoint_dir}/{reason}.pth"
                if os.path.exists(reason_path):
                    os.remove(reason_path) # os.symlink errors if the path exists
                os.symlink(checkpoint_path, reason_path)

        # store the loss history
        print(f"BaselineVGGNet: updating {history_path}")
        pd.DataFrame({
            "epoch": np.arange(epoch + 1) + 1,
            "trn_loss": trn_loss_history,
            "val_loss": val_loss_history,
            "val_acc": val_acc_history,
        }).to_csv(history_path)

    print("BaselineVGGNet: training complete!")
    checkpoint_paths = {
        reason: os.readlink(f"{checkpoint_dir}/{reason}.pth") for reason in [
            "finalEpoch",
            "bestValLoss",
        ]
    } # follow symlinks of all reasons_to_save
    return checkpoint_paths
