"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    main.py
"""

import os
from utils import create_predictions


from train import train


if __name__ == '__main__':
    config_dict = dict()

    config_dict['seed'] = 42
    config_dict['testset_ratio'] = 0.1
    config_dict['validset_ratio'] = 0.1
    config_dict['results_path'] = os.path.join("results")
    config_dict['data_path'] = os.path.join("data", "dataset")
    config_dict['device'] = None
    config_dict['learningrate'] = 1e-3
    config_dict['weight_decay'] = 1e-5 # default is 0
    config_dict['n_updates'] = 50000
    config_dict['batchsize'] = 32
    config_dict['early_stopping_patience'] = 3
    config_dict['use_wandb'] = False

    config_dict['print_train_stats_at'] = 10
    config_dict['print_stats_at'] = 100
    config_dict['plot_at'] = 100
    config_dict['validate_at'] = 100

    network_config = {
        'n_in_channels': 4
    }
    
    config_dict['network_config'] = network_config

    train(**config_dict)
    
    testset_path = os.path.join("data", "challenge_testset.npz")
    state_dict_path = os.path.join(config_dict['results_path'], "best_model.pt")
    save_path = os.path.join(config_dict['results_path'], "testset", "my_submission_name.npz")
    plot_path = os.path.join(config_dict['results_path'], "testset", "plots")

    # Comment out, if predictions are required
    create_predictions(config_dict['network_config'], state_dict_path, testset_path, None, save_path, plot_path, plot_at=20)
