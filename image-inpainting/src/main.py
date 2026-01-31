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
    config_dict['validset_ratio'] = 0.05
    # Get the absolute path based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dict['results_path'] = os.path.join(project_root, "results")
    config_dict['data_path'] = os.path.join(project_root, "data", "dataset")
    config_dict['device'] = None
    config_dict['learningrate'] = 5e-4  # Lower initial LR with warmup
    config_dict['weight_decay'] = 5e-5  # Reduced for more capacity
    config_dict['n_updates'] = 12000  # Extended training for better convergence
    config_dict['batchsize'] = 64  # Reduced for larger model and mixed precision
    config_dict['early_stopping_patience'] = 20  # More patience for complex model
    config_dict['use_wandb'] = False

    config_dict['print_train_stats_at'] = 10
    config_dict['print_stats_at'] = 200
    config_dict['plot_at'] = 500
    config_dict['validate_at'] = 250  # More frequent validation

    network_config = {
        'n_in_channels': 4,
        'base_channels': 52,  # Increased capacity for better feature extraction
        'dropout': 0.15  # Slightly higher dropout for regularization
    }
    
    config_dict['network_config'] = network_config
    
    # Prepare paths for runtime predictions
    testset_path = os.path.join(project_root, "data", "challenge_testset.npz")
    save_path = os.path.join(config_dict['results_path'], "runtime_predictions")
    plot_path_predictions = os.path.join(config_dict['results_path'], "runtime_predictions", "plots")
    
    config_dict['testset_path'] = testset_path
    config_dict['save_path'] = save_path
    config_dict['plot_path_predictions'] = plot_path_predictions
    
    print("="*60)
    print("RUNTIME CONFIGURATION ENABLED")
    print("="*60)
    print("During training, you can modify these parameters by editing:")
    print(f"{os.path.join(config_dict['results_path'], 'runtime_config.json')}")
    print("\nModifiable parameters:")
    print("  - n_updates: Maximum training steps")
    print("  - plot_at: How often to save plots")
    print("  - early_stopping_patience: Patience for early stopping")
    print("  - print_stats_at: How often to print detailed stats")
    print("  - print_train_stats_at: How often to print training loss")
    print("  - validate_at: How often to run validation")
    print("\nRuntime commands (set to true to execute):")
    print("  - save_checkpoint: Save model at current step")
    print("  - run_test_validation: Run validation on final test set")
    print("  - generate_predictions: Generate predictions on challenge testset")
    print("\nChanges will be applied within 5 steps.")
    print("="*60)
    print()

    rmse_value = train(**config_dict)
    
    state_dict_path = os.path.join(config_dict['results_path'], "best_model.pt")
    final_save_path = os.path.join(config_dict['results_path'], "testset", "tikaiz")
    final_plot_path = os.path.join(config_dict['results_path'], "testset", "plots")
    os.makedirs(final_plot_path, exist_ok=True)
    for name in os.listdir(final_plot_path):
        p = os.path.join(final_plot_path, name)
        if os.path.isfile(p) or os.path.islink(p):
            os.unlink(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    # Comment out, if predictions are required
    create_predictions(config_dict['network_config'], state_dict_path, testset_path, None, final_save_path, final_plot_path, plot_at=20, rmse_value=rmse_value)
