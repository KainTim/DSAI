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

    for i in range(5):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            img = data[i:i + 1:, 0:3, :, :]
            img = np.squeeze(img)
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
    """Returnse MSE and RMSE of the model on the provided dataloader"""
    network.eval()
    loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            input_array, target = data
            input_array = input_array.to(device)
            target = target.to(device)

            outputs = network(input_array)

            loss += loss_fn(outputs, target).item()

        loss = loss / len(dataloader)

        network.train()

        return loss, 255.0 * np.sqrt(loss)


def read_compressed_file(file_path: str):
    with np.load(file_path) as data:
        input_arrays = data['input_arrays']
        known_arrays = data['known_arrays']
    return input_arrays, known_arrays


def apply_tta(model, input_tensor, device):
    """
    Apply Test-Time Augmentation for better predictions.
    Averages predictions from original and augmented versions.
    """
    outputs = []
    
    # Original
    out = model(input_tensor)
    outputs.append(out)
    
    # Horizontal flip
    flipped_h = torch.flip(input_tensor, dims=[3])
    out_h = model(flipped_h)
    out_h = torch.flip(out_h, dims=[3])
    outputs.append(out_h)
    
    # Vertical flip
    flipped_v = torch.flip(input_tensor, dims=[2])
    out_v = model(flipped_v)
    out_v = torch.flip(out_v, dims=[2])
    outputs.append(out_v)
    
    # Both flips
    flipped_hv = torch.flip(input_tensor, dims=[2, 3])
    out_hv = model(flipped_hv)
    out_hv = torch.flip(out_hv, dims=[2, 3])
    outputs.append(out_hv)
    
    # Average all predictions
    return torch.stack(outputs, dim=0).mean(dim=0)


def create_predictions(model_config, state_dict_path, testset_path, device, save_path, plot_path, plot_at=20, rmse_value=None, use_tta=True):
    """
    Create predictions with optional Test-Time Augmentation for improved results.
    """

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    model = MyModel(**model_config)
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
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
            input_array = torch.from_numpy(input_arrays[i]).to(device)
            input_tensor = input_array.unsqueeze(0) if input_array.dim() == 3 else input_array
            
            if use_tta:
                output = apply_tta(model, input_tensor, device)
            else:
                output = model(input_tensor)
            
            output = output.cpu().numpy()
            predictions.append(output)

            if (i + 1) % plot_at == 0:
                testset_plot(input_array.cpu().numpy(), output, plot_path, i)

    predictions = np.stack(predictions, axis=0)

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
