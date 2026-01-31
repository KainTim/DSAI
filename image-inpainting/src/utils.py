"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    utils.py
"""

import torch
import numpy as np
import os
from matplotlib import pyplot as plt

from architecture import MyModel


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""

    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    # Only plot up to min(5, batch_size) images
    num_images = min(5, inputs.shape[0])
    
    for i in range(num_images):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            img = data[i, 0:3, :, :]
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update + 1:07d}_{i + 1:02d}.jpg"))

    plt.close(fig)


def testset_plot(input_array, output_array, path, index):
    """Plotting the inputs, targets and predictions to file `path` for testset (no targets available)"""

    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    for ax, data, title in zip(axes, [input_array, output_array], ["Input", "Prediction"]):
        ax.clear()
        ax.set_title(title)
        img = data[0:3, :, :]
        img = np.squeeze(img)
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_axis_off()
    fig.savefig(os.path.join(path, f"testset_{index + 1:07d}.jpg"))

    plt.close(fig)


def evaluate_model(network: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Returns MSE and RMSE of the model on the provided dataloader"""
    # Save training mode and switch to eval
    was_training = network.training
    network.eval()
    
    loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for data in dataloader:
            input_array, target = data
            input_array = input_array.to(device)
            target = target.to(device)
            
            # Check input validity
            if not torch.isfinite(input_array).all() or not torch.isfinite(target).all():
                print(f"Warning: NaN detected in evaluation inputs")
                continue

            outputs = network(input_array)
            
            # Clamp outputs to valid range
            outputs = torch.clamp(outputs, 0.0, 1.0)
            
            # Check for NaN in outputs
            if not torch.isfinite(outputs).all():
                print(f"Warning: NaN detected in model outputs during evaluation")
                continue
            
            batch_loss = loss_fn(outputs, target).item()
            
            # Check for NaN in loss
            if not np.isfinite(batch_loss):
                print(f"Warning: NaN detected in loss during evaluation")
                continue
            
            loss += batch_loss
            num_batches += 1
        
        if num_batches == 0:
            print("Error: No valid batches in evaluation")
            if was_training:
                network.train()
            return float('nan'), float('nan')
        
        loss = loss / num_batches
        rmse = 255.0 * np.sqrt(loss)

        # Restore training mode
        if was_training:
            network.train()

        return loss, rmse


def read_compressed_file(file_path: str):
    with np.load(file_path) as data:
        input_arrays = data['input_arrays']
        known_arrays = data['known_arrays']
    return input_arrays, known_arrays


def create_predictions(model_config, state_dict_path, testset_path, device, save_path, plot_path, plot_at=20, rmse_value=None):
    """
    Here, one might needs to adjust the code based on the used preprocessing
    """

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    model = MyModel(**model_config)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()

    input_arrays, known_arrays = read_compressed_file(testset_path)

    known_arrays = known_arrays.astype(np.float32)

    input_arrays = input_arrays.astype(np.float32) / 255.0

    input_arrays = np.concatenate((input_arrays, known_arrays), axis=1)

    predictions = list()

    with torch.no_grad():
        for i in range(len(input_arrays)):
            print(f"Processing image {i + 1}/{len(input_arrays)}")
            input_array = torch.from_numpy(input_arrays[i]).to(
                device)
            output = model(input_array.unsqueeze(0) if hasattr(input_array, 'dim') and input_array.dim() == 3 else input_array)
            output = output.cpu().numpy()
            predictions.append(output)

            if (i + 1) % plot_at == 0:
                testset_plot(input_array.cpu().numpy(), output, plot_path, i)

    predictions = np.stack(predictions, axis=0)

    # Handle NaN and inf values before conversion
    nan_mask = ~np.isfinite(predictions)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"Warning: Found {nan_count} NaN/Inf values in predictions. Replacing with 0.")
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    
    predictions = (np.clip(predictions, 0, 1) * 255.0).astype(np.uint8)

    data = {
        "predictions": predictions
    }

    # Modify save_path to include RMSE value if provided
    if rmse_value is not None:
        base_path = save_path.rsplit('.npz', 1)[0]
        save_path = f"{base_path}-{rmse_value:.4f}.npz"

    np.savez_compressed(save_path, **data)

    print(f"Predictions saved at {save_path}")
