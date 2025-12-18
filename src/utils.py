import numpy as np
import os

def load_data(function_id):
    """
    Load inputs and outputs from the data folder for a given function_id.
    
    Args:
        function_id (int or str): The ID of the function to load data for.
        
    Returns:
        tuple: (inputs, outputs) loaded from .npy files.
               Returns (None, None) if files are not found.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    function_dir = os.path.join(data_dir, f'function_{function_id}')
    inputs_path = os.path.join(function_dir, 'initial_inputs.npy')
    outputs_path = os.path.join(function_dir, 'initial_outputs.npy')
    
    if not os.path.exists(inputs_path) or not os.path.exists(outputs_path):
        print(f"Data files for function_id {function_id} not found in {function_dir}")
        return None, None
        
    inputs = np.load(inputs_path)
    outputs = np.load(outputs_path)
    
    return inputs, outputs
