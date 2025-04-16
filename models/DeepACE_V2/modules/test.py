import torch
from torch.utils.data import DataLoader
from dataset import Dataset, collate_fn
import os
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import *
from model import DeepACE
import scipy.io

def test_model(model, test_mixture_dir, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    test_dataset = Dataset(test_mixture_dir, None,
                                sample_rate=config['sample_rate'], 
                                segment_length=config['segment_length'], 
                                is_test=True)

    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=config['num_workers'], collate_fn=collate_fn)

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_data_loader, desc='Testing', leave=False)
        for _, (mix, filenames) in enumerate(test_loader_tqdm):
            mix = mix.to(device)

            # Forward pass to get the estimate
            estimate = model(mix)[0].cpu().numpy().transpose(-1, -2)
            estimate[estimate < config['base_level']] = 0


            base_filename = Path(filenames[0]).stem
            cleaned_filename = base_filename.replace("mixed", "")
            output_filename = os.path.join(output_dir, cleaned_filename + "prediction.mat")

            scipy.io.savemat(output_filename, {'pred_lgf': estimate})
                

if __name__ == '__main__':
    # Automatically find all trained models in the 'models/' directory
    model_paths = find_trained_models()
    print(f"Found models: {model_paths}")
    
    parser = argparse.ArgumentParser(description="Test Convdeepace")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Loop through each model and run predictions
    for model_path in model_paths:
        print(f"Processing model: {model_path}")

        # Each model directory is assumed to have a corresponding 'config.yaml'
        model_dir = os.path.dirname(model_path)
        config_path = os.path.join(model_dir, 'config.yaml')

        # Load the configuration file
        config = load_config(config_path)

        # Extract deepace model parameters from the configuration
        deepace_params = config['DeepACE']

        # Initialize the Convdeepace model using parameters from the config file
        model = DeepACE(
            L=deepace_params['L'],
            N=deepace_params['N'],
            P=deepace_params['P'],
            B=deepace_params['B'],
            S=deepace_params['S'],
            H=deepace_params['H'],
            R=deepace_params['R'],
            X=deepace_params['X'],
            M=deepace_params['M'],
            msk_activate=deepace_params['msk_activate'],
            causal=deepace_params['causal']
        ).to(device)

        # Load the saved model weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))


        # Get the directory where the model is saved
        model_dir = os.path.dirname(model_path)

        # Create the predictions directory inside the model folder
        output_dir = os.path.join(model_dir, 'predictions/')
        os.makedirs(output_dir, exist_ok=True)

        # Define directory for the test data (assuming you have this in your config)
        test_dir = config['test_dir']

        # Run the test for this model
        test_model(model, test_dir, output_dir, config)
