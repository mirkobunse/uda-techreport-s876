import numpy as np
import os
import pandas as pd
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax
from sklearn.metrics import zero_one_loss
from torch.autograd import Function
from tqdm.auto import tqdm

class GradientReversalFunction(Function):
    """Gradient reversal layer [ganin2015unsupervised]."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.alpha * grad_output.neg()
        return output, None

class Flatten(nn.Module):
    """Custom layer copied from https://github.com/sbuschjaeger/Pysembles"""
    def __init__(self, store_shape=False):
        super(Flatten, self).__init__()
        self.store_shape = store_shape
    def forward(self, x):
        if self.store_shape:
            self.shape = x.shape
        return x.flatten(1)

# like itertools.cycle but reshuffling a DataLoader instance in each cycle
class ShuffleCycle(object):
    def __init__(self, dataloader):
         self.dataloader = dataloader
         self.iter = iter(self.dataloader)
    def __iter__(self):
         return self
    def __next__(self):
         try:
                 return next(self.iter)
         except StopIteration:
                 self.iter = iter(self.dataloader)
                 return next(self.iter)

class GaninVGGNet(nn.Module):
    """
    This VGGNet extends the baseline [buschjaeger2020onsite] with a UDA component
    based on domain classification with gradient reversal [ganin2015unsupervised].
    """
    def __init__(
            self,
            input_size = (1, 46, 45),
            n_channels = 128,
            depth = 4,
            hidden_size = 512,
            hidden_size_dc = 512,
            num_layers_dc = 2,
            p_dropout = 0.3,
            n_classes = 2
        ):
        super().__init__()
        in_channels = input_size[0]
        feature_extractor = []
        for level in range(depth):
            feature_extractor.extend([
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
        for l in feature_extractor:
            x = l(x)
        lin_size = x.view(1, -1).size()[1]
        feature_extractor = filter(None, feature_extractor)

        # set up the label predictor
        label_predictor = filter(None, [
            Flatten(),
            nn.Linear(lin_size, hidden_size),
            nn.Dropout(p_dropout) if p_dropout > 0 else None,
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        ])

        # set up the domain classifier
        domain_classifier = [ Flatten() ]
        for n in range(num_layers_dc):
            domain_classifier.extend([
                nn.Linear(lin_size if n == 0 else hidden_size_dc, hidden_size_dc),
                nn.Dropout(p_dropout) if p_dropout > 0 else None,
                nn.BatchNorm1d(hidden_size_dc),
                nn.ReLU(),
            ])
        domain_classifier.append(nn.Linear(hidden_size_dc, 2))
        domain_classifier = filter(None, domain_classifier)

        # initialize neural network components from lists of layers
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.label_predictor = nn.Sequential(*label_predictor)
        self.domain_classifier = nn.Sequential(*domain_classifier)

    def forward(self, x, alpha):
        f_out = self.feature_extractor(x)
        l_out = self.label_predictor(f_out)
        rev_f_out = GradientReversalFunction.apply(f_out, alpha)
        d_out = self.domain_classifier(rev_f_out)
        return l_out, d_out


# Ganin et al use dc_weight = 0.1, but do not divide dc_weight/2 like we do to
# compensate for the fact that the domain classifier sees twice the amount of
# the data, as compared to the label predictor. Setting dc_weight = 0.2 thus
# provides the same loss weighting as Ganin et al use.
#
# https://github.com/ddtm/caffe/blob/25fddca8ac00fc0eb7a471a7dd7d7c1d9170d9f1/examples/adaptation/protos/train_val.prototxt#L670
def train_GaninVGGNet(
        model,
        checkpoint_dir,
        X_trn,
        y_trn,
        X_val,
        y_val,
        Z_trn,
        Z_val,
        gamma = 10,
        dc_weight = 0.2,
        n_epochs = 25,
        lr = 1e-4
    ):
    """Train a GaninVGGNet model."""
    print(f"GaninVGGNet: starting the training for {checkpoint_dir}/*.pth")
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

    # dataloader for adaptation (target) data
    ds_Z_trn = torch.utils.data.TensorDataset(
        torch.tensor(Z_trn, dtype=torch.float32).unsqueeze(1)
    )
    dl_Z_trn = torch.utils.data.DataLoader(ds_Z_trn, batch_size=128, shuffle=True)

    # dataloader for validation (target) data
    ds_Z_val = torch.utils.data.TensorDataset(
        torch.tensor(Z_val, dtype=torch.float32).unsqueeze(1))
    dl_Z_val = torch.utils.data.DataLoader(ds_Z_val, batch_size=128, shuffle=True)

    # the loss function and the optimizer
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)

    best_label_loss = np.inf
    best_overall_loss = np.inf
    label_loss_history = []
    domain_loss_history = []
    overall_loss_history = []
    label_acc_history = []
    domain_acc_history = []
    for epoch in range(n_epochs): # loop over the entire dataset multiple times
        model.train()
        running_loss = 0.0
        running_label_loss = 0.0
        running_domain_loss = 0.0
        running_count = 0
        progressbar = tqdm(
            zip(dl_Xy_trn, ShuffleCycle(dl_Z_trn)),
            desc = f"Epoch {epoch+1}/{n_epochs}",
            total = int(len(dl_Xy_trn)),
            ncols = 80
        )

        # progress and alpha value
        p = float(epoch) / n_epochs
        alpha = 2. / (1. + np.exp(-gamma * p)) - 1

        # loop over all batches
        for (X_i, y_i), (Z_i,) in progressbar:
            optimizer.zero_grad() # reset the gradients

            # forward pass of the i-th batch of the source data
            label_pred_s, domain_pred_s = model(X_i.cuda(), alpha)
            label_loss_s = criterion(label_pred_s, y_i.cuda()) # label loss
            domain_loss_s = criterion(
                domain_pred_s,
                torch.zeros(len(y_i)).long().cuda()
            ) # domain loss for the source data

            # forward pass target data
            _, domain_pred_t = model(Z_i.cuda(), alpha)
            domain_loss_t = criterion(
                domain_pred_t,
                torch.ones(len(Z_i)).long().cuda()
            ) # domain loss for the target data

            # sum of losses: backward pass + optimization step
            loss = label_loss_s + dc_weight/2 * (domain_loss_s + domain_loss_t)
            loss.mean().backward()
            optimizer.step()

            running_loss += loss.sum().item()
            running_label_loss += label_loss_s.sum().item()
            running_domain_loss += domain_loss_s.sum().item()
            running_domain_loss += domain_loss_t.sum().item()
            running_count += len(X_i)
            progressbar.set_postfix(
                loss = running_loss / running_count,
                L = running_label_loss / running_count,
                D = running_domain_loss / running_count / 2
            )

        # validation set accuracy with a progress bar
        model.eval() # validate on the hold-out set
        running_loss = 0.0
        running_label_loss = 0.0
        running_domain_loss = 0.0
        running_label_acc = 0.0
        running_domain_acc = 0.0
        running_count = 0
        progressbar = tqdm(
            zip(dl_Xy_val, ShuffleCycle(dl_Z_val)),
            desc = "Validation",
            total = int(len(dl_Xy_val)),
            ncols = 80
        )
        for (X_i, y_i), (Z_i,) in progressbar:
            y_pred, domain_pred_s = model(X_i.cuda(), 1)
            running_label_loss += criterion(y_pred, y_i.cuda()).sum().item()
            y_pred = y_pred.argmax(1).detach().cpu()
            running_label_acc += np.sum(y_pred.numpy() == y_i.numpy())
            running_count += len(X_i)

            # domain predictions for the source validation samples
            running_domain_loss += criterion(
                domain_pred_s,
                torch.zeros(len(y_i)).long().cuda()
            ).sum().item()
            domain_pred_s = domain_pred_s.argmax(1).detach().cpu().numpy()
            running_domain_acc += np.sum(domain_pred_s == np.zeros(len(X_i), dtype=int))

            # predictions for target validation samples
            _, domain_pred_t = model(Z_i.cuda(), 1)
            running_domain_loss += criterion(
                domain_pred_t,
                torch.ones(len(Z_i)).long().cuda()
            ).sum().item()
            domain_pred_t = domain_pred_t.argmax(1).detach().cpu().numpy()
            running_domain_acc += np.sum(domain_pred_t == np.ones(len(Z_i), dtype=int))
        label_loss = running_label_loss / running_count
        domain_loss = running_domain_loss / running_count / 2
        overall_loss = label_loss - dc_weight * domain_loss # combined loss
        label_acc = running_label_acc / running_count
        domain_acc = running_domain_acc / running_count / 2
        label_loss_history.append(label_loss)
        domain_loss_history.append(domain_loss)
        overall_loss_history.append(overall_loss)
        label_acc_history.append(label_acc)
        domain_acc_history.append(domain_acc)
        scheduler.step(overall_loss)
        print(f"GaninVGGNet [epoch {epoch+1}]: val label loss {label_acc}, val domain loss {domain_acc}")

        # collect reasons for creating a checkpoint from this epoch
        reasons_to_save = []
        if (epoch+1) == n_epochs: # save the final epoch
            reasons_to_save.append('finalEpoch')
        if label_loss < best_label_loss:
            reasons_to_save.append('bestLabelLoss')
            best_label_loss = label_loss
        if overall_loss < best_overall_loss:
            reasons_to_save.append('bestOverallLoss')
            best_overall_loss = overall_loss

        # store the model only once
        if len(reasons_to_save) > 0:
            checkpoint_path = f"{checkpoint_dir}/epoch{epoch+1:03d}.pth"
            print(f"GaninVGGNet: saving model at {checkpoint_path} {reasons_to_save}")
            torch.save(model, checkpoint_path)
            for reason in reasons_to_save:
                reason_path = f"{checkpoint_dir}/{reason}.pth"
                if os.path.exists(reason_path):
                    os.remove(reason_path) # os.symlink errors if the path exists
                os.symlink(checkpoint_path, reason_path)

        # store the loss history
        print(f"GaninVGGNet: updating {history_path}")
        pd.DataFrame({
            "epoch": np.arange(epoch + 1) + 1,
            "label_loss": label_loss_history,
            "domain_loss": domain_loss_history,
            "overall_loss": overall_loss_history,
            "label_acc": label_acc_history,
            "domain_acc": domain_acc_history,
        }).to_csv(history_path)

    print("GaninVGGNet: training complete!")
    checkpoint_paths = {
        reason: os.readlink(f"{checkpoint_dir}/{reason}.pth") for reason in [
            "finalEpoch",
            "bestLabelLoss",
            "bestOverallLoss"
        ]
    } # follow symlinks of all reasons_to_save
    return checkpoint_paths
