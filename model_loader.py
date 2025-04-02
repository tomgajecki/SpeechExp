import os
import sys
import importlib.util
import inspect
import torch

def load_model_class(model_dir):
    """
    Load the model class from a directory containing a model.py file.
    
    Args:
        model_dir (str): Path to the directory containing the model.
        
    Returns:
        class: The model class.
    """
    # Clear ALL cached modules that might interfere with loading
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(name in module_name.lower() for name in ['model', 'netblocks', 'ded', 'mask', 'encoder', 'decoder']):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        del sys.modules[module_name]
        print(f"Cleared cached module: {module_name}")
    
    # Get the modules directory
    modules_dir = os.path.join(model_dir, 'modules')
    if not os.path.exists(modules_dir):
        raise ValueError(f"Modules directory not found at {modules_dir}")
    
    # Get the model.py file
    model_file = os.path.join(modules_dir, 'model.py')
    if not os.path.exists(model_file):
        raise ValueError(f"Model file not found at {model_file}")
    
    # Remove any existing paths that might interfere
    sys.path = [p for p in sys.path if not any(name in p.lower() for name in ['deepace_v1', 'deepace_v2', 'modules'])]
    
    # Add the necessary paths in the correct order
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)
        print(f"Added {modules_dir} to Python path")
    
    parent_dir = os.path.dirname(modules_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added {parent_dir} to Python path")
    
    root_dir = os.path.dirname(parent_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        print(f"Added {root_dir} to Python path")
    
    # Import the model module with a unique name based on the directory
    model_name = f"model_{os.path.basename(parent_dir).lower()}"
    spec = importlib.util.spec_from_file_location(model_name, model_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    # List all classes in the module
    model_classes = [cls for name, cls in inspect.getmembers(model_module) 
                    if inspect.isclass(cls) and cls.__module__ == model_module.__name__]
    print(f"Available classes in {model_file}: {[cls.__name__ for cls in model_classes]}")
    
    # Find the main model class
    model_class = None
    
    # First try to find a class that matches the parent directory name
    parent_dir_name = os.path.basename(parent_dir)
    for cls in model_classes:
        if cls.__name__ == parent_dir_name:
            model_class = cls
            break
    
    # If no match found, try common model names
    if model_class is None:
        common_names = ['DeepACE', 'ConvTasNet', 'TasNet']
        for cls in model_classes:
            if cls.__name__ in common_names:
                model_class = cls
                break
    
    # If still no match found, use the first class that inherits from nn.Module
    if model_class is None:
        for cls in model_classes:
            if issubclass(cls, torch.nn.Module):
                model_class = cls
                break
    
    if model_class is None:
        raise ValueError(f"No suitable model class found in {model_file}")
    
    print(f"Selected model class: {model_class.__name__}")
    return model_class 