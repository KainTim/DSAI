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
    config_dict['learningrate'] = 1e-3  # Higher max LR for OneCycleLR
    config_dict['weight_decay'] = 1e-5  # Lower for faster learning
    config_dict['n_updates'] = 3500  # Reduced for fast training
    config_dict['batchsize'] = 96  # Larger batch for speed
    config_dict['early_stopping_patience'] = 3  # Adjusted patience
    config_dict['use_wandb'] = False

    config_dict['print_train_stats_at'] = 10
    config_dict['print_stats_at'] = 100
    config_dict['plot_at'] = 500
    config_dict['validate_at'] = 500  # More frequent validation

    network_config = {
        'n_in_channels': 4,
        'base_channels': 40,  # Reduced for lower complexity
        'dropout': 0.05  # Lower dropout for faster convergence
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
