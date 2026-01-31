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
import json
from torchvision import models
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Subset

import wandb


def load_runtime_config(config_path, current_params):
    """Load runtime configuration from JSON file and update parameters"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                new_config = json.load(f)
            
            # Update modifiable parameters
            updated = False
            modifiable_keys = ['n_updates', 'plot_at', 'early_stopping_patience', 
                             'print_stats_at', 'print_train_stats_at', 'validate_at',
                             'learningrate', 'weight_decay']
            
            for key in modifiable_keys:
                if key in new_config and new_config[key] != current_params.get(key):
                    old_val = current_params.get(key)
                    current_params[key] = new_config[key]
                    print(f"\n[CONFIG UPDATE] {key}: {old_val} -> {new_config[key]}")
                    updated = True
            
            # Check for command flags
            commands = new_config.get('commands', {})
            current_params['commands'] = commands
            
            if updated:
                print("[CONFIG UPDATE] Runtime configuration updated successfully!\n")
    except Exception as e:
        print(f"Warning: Could not load runtime config: {e}")
    
    return current_params


def clear_command_flag(config_path, command_name):
    """Clear a specific command flag after execution"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'commands' in config and command_name in config['commands']:
                config['commands'][command_name] = False
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not clear command flag: {e}")


class RMSELoss(nn.Module):
    """RMSE loss for direct optimization of evaluation metric"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        mse = self.mse(pred, target)
        # Larger epsilon for numerical stability
        rmse = torch.sqrt(mse + 1e-6)
        return rmse


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features for better texture and detail preservation"""
    def __init__(self, device):
        super().__init__()
        # Load pre-trained VGG16 and use specific layers
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Use early and middle layers for perceptual loss
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        
        # Normalization for VGG (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize images for VGG with clamping for stability"""
        # Clamp input to valid range
        x = torch.clamp(x, 0.0, 1.0)
        return (x - self.mean) / (self.std + 1e-8)
    
    def forward(self, pred, target):
        # Clamp inputs to prevent extreme values
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        # Normalize inputs
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Extract features from multiple layers
        pred_f1 = self.slice1(pred)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)
        
        target_f1 = self.slice1(target)
        target_f2 = self.slice2(target_f1)
        target_f3 = self.slice3(target_f2)
        
        # Compute losses at multiple scales
        loss = F.l1_loss(pred_f1, target_f1) + \
               F.l1_loss(pred_f2, target_f2) + \
               F.l1_loss(pred_f3, target_f3)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss optimized for RMSE evaluation with optional perceptual component"""
    def __init__(self, device, use_perceptual=True, perceptual_weight=0.05):
        super().__init__()
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(device)
        # Use MSE instead of RMSE for training (more stable gradients)
        self.mse_loss = nn.MSELoss()
        self.rmse_loss = RMSELoss()  # For logging only
        
        self.perceptual_weight = perceptual_weight
        self.mse_weight = 1.0 - perceptual_weight
    
    def forward(self, pred, target):
        # Clamp predictions to valid range
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        # Check for NaN in inputs
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            print("Warning: NaN detected in loss inputs")
            return (torch.tensor(float('nan'), device=pred.device),) * 4
        
        # Primary loss: MSE (equivalent to RMSE but more stable)
        mse = self.mse_loss(pred, target)
        rmse = self.rmse_loss(pred, target)  # For logging
        
        if self.use_perceptual:
            # Optional small perceptual component for texture quality
            perceptual = self.perceptual_loss(pred, target)
            # Check perceptual loss validity
            if not torch.isfinite(perceptual):
                perceptual = torch.tensor(0.0, device=pred.device)
            total_loss = self.mse_weight * mse + self.perceptual_weight * perceptual
        else:
            # Pure MSE optimization
            perceptual = torch.tensor(0.0, device=pred.device)
            total_loss = mse
        
        # Validate loss is not NaN or Inf
        if not torch.isfinite(total_loss):
            # Return MSE only as fallback
            total_loss = mse
            if not torch.isfinite(total_loss):
                print("Warning: MSE is NaN")
                return (torch.tensor(float('nan'), device=pred.device),) * 4
        
        return total_loss, perceptual, mse, rmse


def train(seed, testset_ratio, validset_ratio, data_path, results_path, early_stopping_patience, device, learningrate,
          weight_decay, n_updates, use_wandb, print_train_stats_at, print_stats_at, plot_at, validate_at, batchsize,
          network_config: dict, testset_path=None, save_path=None, plot_path_predictions=None):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)
    
    # Enable mixed precision training for memory efficiency
    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler = torch.amp.GradScaler('cuda', init_scale=2048.0, growth_interval=100)
    else:
        scaler = None

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

    # defining the loss - Optimized for RMSE evaluation
    # Set use_perceptual=False for pure MSE training, or keep True with 5% weight for texture quality
    # TEMPORARILY DISABLED due to NaN issues - re-enable once training is stable
    combined_loss = CombinedLoss(device, use_perceptual=False, perceptual_weight=0.0).to(device)
    mse_loss = torch.nn.MSELoss()  # Keep for evaluation

    # defining the optimizer with AdamW for better weight decay handling
    optimizer = torch.optim.AdamW(network.parameters(), lr=learningrate, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Learning rate warmup
    warmup_steps = min(1000, n_updates // 10)
    
    # Cosine annealing with warm restarts for long training
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=n_updates//4, T_mult=1, eta_min=learningrate/100
    )
    
    # Warmup scheduler
    def get_lr_scale(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    if use_wandb:
        wandb.watch(network, mse_loss, log="all", log_freq=10)

    i = 0
    counter = 0
    best_validation_loss = np.inf
    loss_list = []

    saved_model_path = os.path.join(results_path, "best_model.pt")
    
    # Save runtime configuration to JSON file for dynamic updates
    config_json_path = os.path.join(results_path, "runtime_config.json")
    runtime_params = {
        'learningrate': learningrate,
        'weight_decay': weight_decay,
        'n_updates': n_updates,
        'plot_at': plot_at,
        'early_stopping_patience': early_stopping_patience,
        'print_stats_at': print_stats_at,
        'print_train_stats_at': print_train_stats_at,
        'validate_at': validate_at,
        'commands': {
            'save_checkpoint': False,
            'run_test_validation': False,
            'generate_predictions': False
        }
    }
    
    with open(config_json_path, 'w') as f:
        json.dump(runtime_params, f, indent=2)
    
    print(f"Started training on device {device}")
    print(f"Runtime config saved to: {config_json_path}")
    print(f"You can modify this file during training to change parameters dynamically!")
    print(f"Set command flags to true to trigger actions (save_checkpoint, run_test_validation, generate_predictions)\n")

    while i < n_updates:

        for input, target in dataloader_train:

            input, target = input.to(device), target.to(device)
            
            # Check for runtime config updates every 50 steps
            if i % 50 == 0 and i > 0:
                runtime_params = load_runtime_config(config_json_path, runtime_params)
                n_updates = runtime_params['n_updates']
                plot_at = runtime_params['plot_at']
                early_stopping_patience = runtime_params['early_stopping_patience']
                print_stats_at = runtime_params['print_stats_at']
                print_train_stats_at = runtime_params['print_train_stats_at']
                validate_at = runtime_params['validate_at']
                
                # Update optimizer parameters if changed
                if 'learningrate' in runtime_params:
                    new_lr = runtime_params['learningrate']
                    current_lr = optimizer.param_groups[0]['lr']
                    if abs(new_lr - current_lr) > 1e-10:  # Float comparison with tolerance
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                
                if 'weight_decay' in runtime_params:
                    new_wd = runtime_params['weight_decay']
                    current_wd = optimizer.param_groups[0]['weight_decay']
                    if abs(new_wd - current_wd) > 1e-10:  # Float comparison with tolerance
                        for param_group in optimizer.param_groups:
                            param_group['weight_decay'] = new_wd
                
                # Execute runtime commands
                commands = runtime_params.get('commands', {})
                
                # Command: Save checkpoint
                if commands.get('save_checkpoint', False):
                    checkpoint_path = os.path.join(results_path, f"checkpoint_step_{i}.pt")
                    torch.save(network.state_dict(), checkpoint_path)
                    print(f"\n[COMMAND] Checkpoint saved to: {checkpoint_path}\n")
                    clear_command_flag(config_json_path, 'save_checkpoint')
                
                # Command: Generate predictions
                if commands.get('generate_predictions', False) and testset_path is not None:
                    print(f"\n[COMMAND] Generating predictions at step {i}...")
                    try:
                        from utils import create_predictions
                        pred_save_path = save_path or os.path.join(results_path, "runtime_predictions", f"step_{i}")
                        pred_plot_path = plot_path_predictions or os.path.join(results_path, "runtime_predictions", "plots", f"step_{i}")
                        os.makedirs(pred_plot_path, exist_ok=True)
                        
                        # Save current state temporarily
                        temp_state_path = os.path.join(results_path, f"temp_state_step_{i}.pt")
                        torch.save(network.state_dict(), temp_state_path)
                        
                        # Generate predictions
                        create_predictions(network_config, temp_state_path, testset_path, None, 
                                         pred_save_path, pred_plot_path, plot_at=20, rmse_value=None)
                        
                        print(f"[COMMAND] Predictions saved to: {pred_save_path}")
                        print(f"[COMMAND] Plots saved to: {pred_plot_path}\n")
                        
                        # Clean up temp file
                        if os.path.exists(temp_state_path):
                            os.remove(temp_state_path)
                    except Exception as e:
                        print(f"[COMMAND] Error generating predictions: {e}\n")
                    
                    network.train()
                    clear_command_flag(config_json_path, 'generate_predictions')
                
                # Command: Run test validation
                if commands.get('run_test_validation', False):
                    print(f"\n[COMMAND] Running test set validation at step {i}...")
                    network.eval()
                    test_loss, test_rmse = evaluate_model(network, dataloader_test, mse_loss, device)
                    print(f"[COMMAND] Test Loss: {test_loss:.6f}, Test RMSE: {test_rmse:.6f}\n")
                    network.train()
                    clear_command_flag(config_json_path, 'run_test_validation')

            if (i + 1) % print_train_stats_at == 0:
                print(f'Update Step {i + 1} of {n_updates}: Current loss: {loss_list[-1]}')

            optimizer.zero_grad()

            # Mixed precision training for memory efficiency
            if use_amp:
                with torch.amp.autocast('cuda'):
                    output = network(input)
                    total_loss, perceptual, mse, rmse = combined_loss(output, target)
                
                # Check for NaN before backward
                if not torch.isfinite(total_loss):
                    continue
                
                scaler.scale(total_loss).backward()
                
                # Unscale and check gradients
                scaler.unscale_(optimizer)
                
                # Check for NaN in gradients
                has_nan = False
                for name, param in network.named_parameters():
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            print(f"NaN gradient detected in {name}")
                            has_nan = True
                            break
                
                if has_nan:
                    print(f"Skipping step {i+1}: NaN gradients detected")
                    optimizer.zero_grad()
                    scaler.update()
                    # Reset scaler if NaN persists
                    if (i + 1) % 10 == 0:
                        scaler = torch.amp.GradScaler('cuda', init_scale=2048.0, growth_interval=100)
                    continue
                
                # More aggressive gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                
                # Skip update if gradient norm is too large
                if grad_norm > 100.0:
                    print(f"Skipping step {i+1}: Gradient norm too large: {grad_norm:.2f}")
                    optimizer.zero_grad()
                    scaler.update()
                    continue
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = network(input)
                total_loss, perceptual, mse, rmse = combined_loss(output, target)
                
                # Check for NaN before backward
                if not torch.isfinite(total_loss):
                    print(f"Skipping step {i+1}: NaN or Inf loss detected")
                    continue
                
                total_loss.backward()
                
                # Check for NaN in gradients
                has_nan = False
                for name, param in network.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"NaN gradient detected in {name}")
                        has_nan = True
                        break
                
                if has_nan:
                    print(f"Skipping step {i+1}: NaN gradients detected")
                    optimizer.zero_grad()
                    continue
                
                # More aggressive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                
                if grad_norm > 100.0:
                    print(f"Skipping step {i+1}: Gradient norm too large: {grad_norm:.2f}")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
            
            # Apply learning rate scheduling with warmup
            lr_scale = get_lr_scale(i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learningrate * lr_scale
            
            if i >= warmup_steps:
                scheduler_main.step()

            loss_list.append(total_loss.item())

            # writing the stats to wandb
            if use_wandb and (i+1) % print_stats_at == 0:
                wandb.log({
                    "training/loss_total": total_loss.item(),
                    "training/loss_mse": mse.item(),
                    "training/loss_rmse": rmse.item(),
                    "training/loss_perceptual": perceptual.item() if isinstance(perceptual, torch.Tensor) else perceptual,
                    "training/learning_rate": optimizer.param_groups[0]['lr']
                }, step=i)

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
