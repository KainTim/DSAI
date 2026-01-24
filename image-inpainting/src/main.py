"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    main.py
"""

import os
from utils import create_predictions


from train import train
import shutil


if __name__ == '__main__':
    config_dict = dict()

    config_dict['seed'] = 42
    config_dict['testset_ratio'] = 0.1
    config_dict['validset_ratio'] = 0.1
    # Get the absolute path based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dict['results_path'] = os.path.join(project_root, "results")
    config_dict['data_path'] = os.path.join(project_root, "data", "dataset")
    config_dict['device'] = None
    config_dict['learningrate'] = 2e-4  # Slightly lower for more stable training
    config_dict['weight_decay'] = 5e-5  # Reduced for less aggressive regularization
    config_dict['n_updates'] = 8000  # More updates for better convergence
    config_dict['batchsize'] = 12  # Larger batch for more stable gradients
    config_dict['early_stopping_patience'] = 15  # More patience for complex model
    config_dict['use_wandb'] = False

    config_dict['print_train_stats_at'] = 10
    config_dict['print_stats_at'] = 100
    config_dict['plot_at'] = 400
    config_dict['validate_at'] = 200  # Validate frequently but not too often

    network_config = {
        'n_in_channels': 4,
        'base_channels': 32,  # Smaller base for efficiency, depth compensates
        'dropout': 0.15  # Slightly more regularization with augmentation
    }
    
    config_dict['network_config'] = network_config

    rmse_value = train(**config_dict)
    
    testset_path = os.path.join(project_root, "data", "challenge_testset.npz")
    state_dict_path = os.path.join(config_dict['results_path'], "best_model.pt")
    save_path = os.path.join(config_dict['results_path'], "testset", "tikaiz")
    plot_path = os.path.join(config_dict['results_path'], "testset", "plots")
    os.makedirs(plot_path, exist_ok=True)
    for name in os.listdir(plot_path):
        p = os.path.join(plot_path, name)
        if os.path.isfile(p) or os.path.islink(p):
            os.unlink(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    # Comment out, if predictions are required
    create_predictions(config_dict['network_config'], state_dict_path, testset_path, None, save_path, plot_path, plot_at=20, rmse_value=rmse_value)
