import argparse
import os
import torch
import torchaudio
import numpy as np
import importlib
import yaml
import json
import sys
from pathlib import Path

def load_config(config_path):
    """Load configuration from a YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def load_audio(audio_path):
    """Load audio file and convert to tensor."""
    if isinstance(audio_path, str):
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform
    elif isinstance(audio_path, np.ndarray):
        return torch.from_numpy(audio_path)
    elif torch.is_tensor(audio_path):
        return audio_path
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_path)}")

def get_model_class(model_type, modules_dir):
    """Dynamically import the model class based on model type."""
    try:
        print(f"Attempting to load model class for type: {model_type}")
        print(f"Modules directory: {modules_dir}")
        
        # Add the modules directory to the Python path
        if modules_dir not in sys.path:
            sys.path.append(modules_dir)
            print(f"Added {modules_dir} to Python path")
        
        # Get the parent directory name (e.g., DeepACEv1, DeepACEv2)
        parent_dir = os.path.basename(os.path.dirname(modules_dir))
        print(f"Parent directory: {parent_dir}")
        
        # List all Python files in the modules directory
        python_files = [f for f in os.listdir(modules_dir) if f.endswith('.py')]
        print(f"Python files in modules directory: {python_files}")
        
        # Try to import the model module
        try:
            print("Attempting to import 'model' module")
            model_module = importlib.import_module("model")
            print("Successfully imported 'model' module")
            
            # List all attributes in the model module
            module_attrs = dir(model_module)
            print(f"Attributes in model module: {module_attrs}")
            
            # Try to find the model class
            for attr_name in module_attrs:
                attr = getattr(model_module, attr_name)
                if isinstance(attr, type) and attr_name not in ['torch', 'nn', 'np', 'os', 'sys']:
                    print(f"Found potential model class: {attr_name}")
            
            # Try specific class names
            for class_name in [parent_dir, model_type, "DeepACE", "tasnet"]:
                if hasattr(model_module, class_name):
                    print(f"Found model class: {class_name}")
                    return getattr(model_module, class_name)
            
            # If no specific class is found, return the module itself
            print("No specific class found, returning module")
            return model_module
            
        except ImportError as e:
            print(f"Failed to import 'model' module: {e}")
            
            # Try to import a specific file if it exists
            for py_file in python_files:
                if py_file != "__init__.py":
                    try:
                        module_name = os.path.splitext(py_file)[0]
                        print(f"Attempting to import '{module_name}' module")
                        specific_module = importlib.import_module(module_name)
                        print(f"Successfully imported '{module_name}' module")
                        
                        # List all attributes in the specific module
                        module_attrs = dir(specific_module)
                        print(f"Attributes in {module_name} module: {module_attrs}")
                        
                        # Try to find the model class
                        for attr_name in module_attrs:
                            attr = getattr(specific_module, attr_name)
                            if isinstance(attr, type) and attr_name not in ['torch', 'nn', 'np', 'os', 'sys']:
                                print(f"Found potential model class: {attr_name}")
                        
                        # Try specific class names
                        for class_name in [parent_dir, model_type, "DeepACE", "tasnet"]:
                            if hasattr(specific_module, class_name):
                                print(f"Found model class: {class_name}")
                                return getattr(specific_module, class_name)
                        
                        # If no specific class is found, return the module itself
                        print(f"No specific class found in {module_name}, returning module")
                        return specific_module
                        
                    except ImportError as e:
                        print(f"Failed to import '{module_name}' module: {e}")
            
            # If all direct imports fail, try to import from models directory
            try:
                # Use the full path to import the specific model
                module_path = f"models.{parent_dir}.modules.model"
                print(f"Trying to import from {module_path}")
                module = importlib.import_module(module_path)
                print(f"Successfully imported from {module_path}")
                
                # List all attributes in the module
                module_attrs = dir(module)
                print(f"Attributes in {module_path}: {module_attrs}")
                
                # Try specific class names
                for class_name in [parent_dir, model_type, "DeepACE", "tasnet"]:
                    if hasattr(module, class_name):
                        print(f"Found model class: {class_name}")
                        return getattr(module, class_name)
                
                print("No specific class found, returning module")
                return module
                
            except ImportError as e:
                print(f"Failed to import from models directory: {e}")
                raise ImportError(f"Could not find model class for {model_type} in {modules_dir}")
                
    except Exception as e:
        print(f"Error in get_model_class: {e}")
        import traceback
        traceback.print_exc()
        raise ImportError(f"Error importing model class: {e}")

def find_model_files(model_dir):
    """Find model files in the checkpoint directory."""
    checkpoint_dir = os.path.join(model_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for .pth files in the checkpoint directory
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No .pth model files found in {checkpoint_dir}")
    
    # Use the first .pth file found
    model_path = os.path.join(checkpoint_dir, model_files[0])
    return model_path, model_files[0]

def find_config_file(model_dir, model_name=None):
    """Find configuration file in the checkpoint directory first, then modules directory."""
    # First try to find config.yaml in the checkpoint directory
    checkpoint_dir = os.path.join(model_dir, "checkpoint")
    if os.path.exists(checkpoint_dir):
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if os.path.exists(config_path):
            return config_path
        
        # If model_name is provided, try to find a config file with a similar name in checkpoint
        if model_name:
            model_base_name = os.path.splitext(model_name)[0]
            config_path = os.path.join(checkpoint_dir, f"{model_base_name}.yaml")
            if os.path.exists(config_path):
                return config_path
    
    # If not found in checkpoint, try the modules directory
    modules_dir = os.path.join(model_dir, "modules")
    if os.path.exists(modules_dir):
        # Try config.yaml in modules
        config_path = os.path.join(modules_dir, "config.yaml")
        if os.path.exists(config_path):
            return config_path
        
        # If model_name is provided, try to find a config file with a similar name in modules
        if model_name:
            model_base_name = os.path.splitext(model_name)[0]
            config_path = os.path.join(modules_dir, f"{model_base_name}.yaml")
            if os.path.exists(config_path):
                return config_path
        
        # Try to find any yaml file in the modules directory
        yaml_files = [f for f in os.listdir(modules_dir) if f.endswith('.yaml') or f.endswith('.yml')]
        if yaml_files:
            return os.path.join(modules_dir, yaml_files[0])
    
    raise FileNotFoundError(f"No configuration file found in {model_dir}/checkpoint or {model_dir}/modules")

def run_inference(waveform, model_path=None, config_path=None, model_type=None, model_dir=None):
    """
    Run inference with the specified model.
    
    Args:
        waveform: Input audio waveform (tensor, numpy array, or path to audio file)
        model_path: Path to the model weights file (optional if model_dir is provided)
        config_path: Path to the model configuration file (optional if model_dir is provided)
        model_type: Type of model to use (if None, will try to infer from model_dir)
        model_dir: Directory containing the model (optional if model_path and config_path are provided)
    
    Returns:
        Processed audio waveform
    """
    try:
        # Load and preprocess audio
        audio = load_audio(waveform)
        
        # Convert to float
        audio = audio.float()
        
        # If model_dir is provided, find model and config files
        if model_dir:
            try:
                model_path, model_name = find_model_files(model_dir)
                config_path = find_config_file(model_dir, model_name)
                
                # If model_type is not provided, try to infer from model_dir
                if model_type is None:
                    model_type = os.path.basename(model_dir)
            except Exception as e:
                raise ValueError(f"Error finding model files: {e}")
        
        # Load configuration
        config = load_config(config_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear any existing model from memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        
        # Import the model loader
        from model_loader import load_model_class
        
        # Load the model class directly from the model directory
        model_class = load_model_class(model_dir)
        
        # Initialize model with parameters from config
        model_params = None
        for key in ['DeepACE', 'tasnet', model_type]:
            if key in config:
                model_params = config[key]
                break
        
        if model_params is None:
            raise ValueError(f"No model parameters found in config for model type: {model_type}")
        
        # Print the model parameters for debugging
        print(f"Model parameters: {model_params}")
        
        # Initialize model
        try:
            model = model_class(**model_params).to(device)
        except TypeError as e:
            print(f"Error initializing model: {e}")
            print(f"Model class: {model_class}")
            print(f"Model parameters: {model_params}")
            raise
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set model to evaluation mode
        model.eval()
        
        # Ensure audio has the correct shape (batch, channel, time)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            output = model(audio)
            
            # Handle different output formats
            if isinstance(output, tuple):
                # Some models return a tuple (output, _)
                output = output[0]
        
        # Convert output to numpy array
        output = output.cpu().numpy()     
        # Clean up to prevent memory leaks
        del model
        del audio
        gc.collect()
        torch.cuda.empty_cache()  
        
        return output[0]
    

    except Exception as e:
        import traceback
        print(f"Error in run_inference: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file or directory")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--output", type=str, help="Path to save the output audio")
    parser.add_argument("--model_type", type=str, help="Type of model to use")
    
    args = parser.parse_args()
    
    # Run inference
    output = run_inference(args.audio, model_dir=args.model_dir, model_type=args.model_type)
    
    # Save output if specified
    if args.output:
        import scipy.io.wavfile as wavfile
        wavfile.write(args.output, 16000, output)  # Assuming 16kHz sample rate
        print(f"Output saved to {args.output}")
    else:
        print("Output shape:", output.shape) 