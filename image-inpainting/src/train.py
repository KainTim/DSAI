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
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import wandb


def gaussian_kernel(window_size=11, sigma=1.5):
    """Create a Gaussian kernel for SSIM computation"""
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()
    kernel_2d = kernel.unsqueeze(1) * kernel.unsqueeze(0)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss for perceptual quality"""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        kernel = gaussian_kernel(window_size, sigma)
        self.register_buffer('kernel', kernel)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def forward(self, pred, target):
        # Apply to each channel
        channels = pred.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        
        mu_pred = F.conv2d(pred, kernel, padding=self.window_size // 2, groups=channels)
        mu_target = F.conv2d(target, kernel, padding=self.window_size // 2, groups=channels)
        
        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=self.window_size // 2, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, kernel, padding=self.window_size // 2, groups=channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, kernel, padding=self.window_size // 2, groups=channels) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
               ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))
        
        return 1 - ssim.mean()


class CombinedLoss(nn.Module):
    """Combined loss: MSE + L1 + SSIM + Edge for comprehensive image reconstruction"""
    def __init__(self, mse_weight=1.0, l1_weight=0.5, edge_weight=0.15, ssim_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=7)
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))
    
    def edge_loss(self, pred, target):
        """Compute edge-aware loss using Sobel filters"""
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        
        edge_loss = self.l1(pred_edge_x, target_edge_x) + self.l1(pred_edge_y, target_edge_y)
        return edge_loss
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        edge_loss = self.edge_loss(pred, target)
        ssim_loss = self.ssim(pred, target)
        
        total_loss = (self.mse_weight * mse_loss + 
                      self.l1_weight * l1_loss + 
                      self.edge_weight * edge_loss +
                      self.ssim_weight * ssim_loss)
        return total_loss


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

    # defining the loss - combined loss for better reconstruction
    combined_loss = CombinedLoss(mse_weight=1.0, l1_weight=0.5, edge_weight=0.15, ssim_weight=0.3).to(device)
    mse_loss = torch.nn.MSELoss()  # Keep for evaluation

    # defining the optimizer with AdamW for better weight decay handling
    optimizer = torch.optim.AdamW(network.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    # Learning rate scheduler for better convergence
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

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

            output = network(input)

            loss = combined_loss(output, target)

            loss.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(i + len(loss_list) / len(dataloader_train))

            loss_list.append(loss.item())

            # writing the stats to wandb
            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({"training/loss_per_batch": loss.item()}, step=i)

            # plotting
            if (i + 1) % plot_at == 0:
                print(f"Plotting images, current update {i + 1}")
                plot(input.cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy(), plotpath, i)

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
