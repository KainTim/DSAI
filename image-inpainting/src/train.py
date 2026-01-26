"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    train.py
"""

import datasets
from architecture import MyModel
from utils import plot, evaluate_model

import torch
import torch.nn as nn
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data import Subset

import wandb


class RMSELoss(nn.Module):
    """RMSE loss for direct optimization of evaluation metric"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        mse = self.mse(pred, target)
        rmse = torch.sqrt(mse + 1e-8)  # Add epsilon for numerical stability
        return rmse


def train(seed, testset_ratio, validset_ratio, data_path, results_path, early_stopping_patience, device, learningrate,
          weight_decay, n_updates, use_wandb, print_train_stats_at, print_stats_at, plot_at, validate_at, batchsize,
          network_config: dict):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)
    
    # Enable mixed precision training for memory efficiency
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    if use_wandb:
        wandb.login()
        wandb.init(project="image_inpainting", config={
            "learning_rate": learningrate,
            "weight_decay": weight_decay,
            "n_updates": n_updates,
            "batch_size": batchsize,
            "validation_ratio": validset_ratio,
            "testset_ratio": testset_ratio,
            "early_stopping_patience": early_stopping_patience,
        })

    # Prepare a path to plot to
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)

    image_dataset = datasets.ImageDataset(datafolder=data_path)

    n_total = len(image_dataset)
    n_test = int(n_total * testset_ratio)
    n_valid = int(n_total * validset_ratio)
    n_train = n_total - n_test - n_valid
    indices = np.random.permutation(n_total)
    dataset_train = Subset(image_dataset, indices=indices[0:n_train])
    dataset_valid = Subset(image_dataset, indices=indices[n_train:n_train + n_valid])
    dataset_test = Subset(image_dataset, indices=indices[n_train + n_valid:n_total])

    assert len(image_dataset) == len(dataset_train) + len(dataset_test) + len(dataset_valid)

    del image_dataset

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batchsize,
                                  num_workers=0, shuffle=True)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1,
                                  num_workers=0, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1,
                                 num_workers=0, shuffle=False)

    # initializing the model
    network = MyModel(**network_config)
    network.to(device)
    network.train()

    # defining the loss - RMSE for direct optimization of evaluation metric
    rmse_loss = RMSELoss().to(device)
    mse_loss = torch.nn.MSELoss()  # Keep for evaluation

    # defining the optimizer with AdamW for better weight decay handling
    optimizer = torch.optim.AdamW(network.parameters(), lr=learningrate, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Cosine annealing with warm restarts for long training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=n_updates//4, T_mult=1, eta_min=learningrate/100
    )

    if use_wandb:
        wandb.watch(network, mse_loss, log="all", log_freq=10)

    i = 0
    counter = 0
    best_validation_loss = np.inf
    loss_list = []

    saved_model_path = os.path.join(results_path, "best_model.pt")

    print(f"Started training on device {device}")

    while i < n_updates:

        for input, target in dataloader_train:

            input, target = input.to(device), target.to(device)

            if (i + 1) % print_train_stats_at == 0:
                print(f'Update Step {i + 1} of {n_updates}: Current loss: {loss_list[-1]}')

            optimizer.zero_grad()

            # Mixed precision training for memory efficiency
            if use_amp:
                with torch.amp.autocast('cuda'):
                    output = network(input)
                    loss = rmse_loss(output, target)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping for training stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = network(input)
                loss = rmse_loss(output, target)
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            scheduler.step()

            loss_list.append(loss.item())

            # writing the stats to wandb
            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({"training/loss_per_batch": loss.item()}, step=i)

            # plotting
            if (i + 1) % plot_at == 0:
                print(f"Plotting images, current update {i + 1}")
                # Convert to float32 for matplotlib compatibility
                plot(input.float().cpu().numpy(), 
                     target.detach().float().cpu().numpy(), 
                     output.detach().float().cpu().numpy(), 
                     plotpath, i)

            # evaluating model every validate_at sample
            if (i + 1) % validate_at == 0:
                print(f"Evaluation of the model:")
                val_loss, val_rmse = evaluate_model(network, dataloader_valid, mse_loss, device)
                print(f"val_loss: {val_loss}")
                print(f"val_RMSE: {val_rmse}")

                if use_wandb:
                    wandb.log({"validation/loss": val_loss,
                               "validation/RMSE": val_rmse}, step=i)
                    # wandb histogram

                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(network.state_dict(), saved_model_path)
                    print(f"Saved new best model with val_loss: {best_validation_loss}")
                    counter = 0
                else:
                    counter += 1

            if counter >= early_stopping_patience:
                print("Stopped training because of early stopping")
                i = n_updates
                break

            i += 1
            if i >= n_updates:
                print("Finished training because maximum number of updates reached")
                break

    print("Evaluating the self-defined testset")
    network.load_state_dict(torch.load(saved_model_path))
    testset_loss, testset_rmse = evaluate_model(network=network, dataloader=dataloader_test, loss_fn=mse_loss,
                                                device=device)

    print(f'testset_loss of model: {testset_loss}, RMSE = {testset_rmse}')

    if use_wandb:
        wandb.summary["testset/loss"] = testset_loss
        wandb.summary["testset/RMSE"] = testset_rmse
        wandb.finish()

    return testset_rmse
