import numpy as np
import os
import ast

# ==========================================
# CONFIGURATION: CHANGE THIS EVERY WEEK
# ==========================================
CURRENT_WEEK = "week1"
# This tells the script to look in: data/update/week1/
# ==========================================

def get_update_paths():
    """Generates paths based on the CURRENT_WEEK variable."""
    base_dir = os.path.join('data', 'update', CURRENT_WEEK)
    return {
        'inputs': os.path.join(base_dir, 'inputs.txt'),
        'outputs': os.path.join(base_dir, 'outputs.txt')
    }

def update_function_data(func_id, new_x, new_y):
    """
    Standard update logic: loads .npy files, appends new row, saves.
    """
    base_path = f'data/function_{func_id}'
    inputs_path = f'{base_path}/initial_inputs.npy'
    outputs_path = f'{base_path}/initial_outputs.npy'

    try:
        # 1. Load existing data
        current_X = np.load(inputs_path)
        current_Y = np.load(outputs_path)

        # 2. Prepare new data
        # Ensure new_x is 2D (1 sample, N features)
        new_x_arr = np.array(new_x).reshape(1, -1)

        # Ensure new_y is correct shape
        # If Y is a scalar float, make it 1D array. If existing Y is 2D, make new Y 2D.
        new_y_arr = np.array([new_y])
        if current_Y.ndim > 1:
            new_y_arr = new_y_arr.reshape(1, -1)

        # 3. Validation
        if new_x_arr.shape[1] != current_X.shape[1]:
            print(f"❌ Func {func_id} Error: Dimension mismatch (New: {new_x_arr.shape[1]}, Old: {current_X.shape[1]})")
            return

        # 4. Append data
        updated_X = np.vstack((current_X, new_x_arr))

        if current_Y.ndim == 1:
            updated_Y = np.append(current_Y, new_y)
        else:
            updated_Y = np.vstack((current_Y, new_y_arr))

        # 5. Save
        np.save(inputs_path, updated_X)
        np.save(outputs_path, updated_Y)

        print(f"✅ Function {func_id} updated. (Total points: {len(updated_X)})")

    except FileNotFoundError:
        print(f"❌ Error: Database files not found for Function {func_id}")
    except Exception as e:
        print(f"❌ Error processing Function {func_id}: {e}")

def parse_text_data(file_path):
    """
    Reads a text file containing Python-style lists/arrays and converts them
    to actual Python objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read().strip()

    # Clean up any potential copy-paste artifacts like "" if present
    if "source:" in content:
        start_index = content.find('[array') if '[array' in content else content.find('[np')
        if start_index != -1:
            content = content[start_index:]

    # Define a safe context for eval() to understand 'array' and 'float64'
    # This matches the format in your text files: [array([...]), ...]
    context = {
        'array': np.array,
        'np': np,
        'float64': np.float64
    }

    return eval(content, context)

def run_weekly_update():
    paths = get_update_paths()

    print(f"📂 Looking for updates in: data/update/{CURRENT_WEEK}/")

    try:
        # Parse the text files
        inputs_list = parse_text_data(paths['inputs'])
        outputs_list = parse_text_data(paths['outputs'])

        if len(inputs_list) != 8 or len(outputs_list) != 8:
            print(f"⚠️ Warning: Expected 8 functions. Found {len(inputs_list)} inputs and {len(outputs_list)} outputs.")

        print("--- Starting Update ---")

        # Iterate through the lists and update each function
        for i in range(len(inputs_list)):
            func_id = i + 1
            x_data = inputs_list[i]
            y_data = outputs_list[i]

            # Handle cases where y_data might be wrapped in a numpy scalar
            if hasattr(y_data, 'item'):
                y_data = y_data.item()

            update_function_data(func_id, x_data, y_data)

        print("--- Update Complete ---")

    except FileNotFoundError as e:
        print(f"❌ Critical Error: {e}")
        print("Please ensure you have created the folder and files exactly as shown in the screenshot.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    run_weekly_update()