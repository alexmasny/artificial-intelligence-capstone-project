import numpy as np
import os
import re
import glob


def get_sorted_weeks():
    """
    Finds all week directories in data/update/ and returns them sorted by week number.
    """
    base_dir = os.path.join("data", "update")
    week_dirs = glob.glob(os.path.join(base_dir, "week*"))

    # Sort by the integer number in the folder name (e.g., "week1", "week2")
    def extract_week_num(d):
        match = re.search(r"week(\d+)", d)
        return int(match.group(1)) if match else 0

    return sorted(week_dirs, key=extract_week_num)


def parse_text_data(file_path):
    """
    Reads a text file containing Python-style lists/arrays and converts them
    to actual Python objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        content = f.read().strip()

    # Clean up any potential copy-paste artifacts like "" if present
    if "source:" in content:
        start_index = (
            content.find("[array") if "[array" in content else content.find("[np")
        )
        if start_index != -1:
            content = content[start_index:]

    # Define a safe context for eval() to understand 'array' and 'float64'
    # This matches the format in your text files: [array([...]), ...]
    context = {"array": np.array, "np": np, "float64": np.float64}

    return eval(content, context)


def load_initial_data(func_id):
    """
    Loads the initial data for a given function.
    """
    base_path = f"data/initial_data/function_{func_id}"
    inputs_path = f"{base_path}/initial_inputs.npy"
    outputs_path = f"{base_path}/initial_outputs.npy"

    if not os.path.exists(inputs_path) or not os.path.exists(outputs_path):
        print(f"⚠️ Initial data not found for Function {func_id} at {base_path}")
        return None, None

    return np.load(inputs_path), np.load(outputs_path)


def save_processed_data(func_id, X, Y):
    """
    Saves the aggregated data to data/processed.
    """
    output_dir = f"data/processed/function_{func_id}"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "initial_inputs.npy"), X)
    np.save(os.path.join(output_dir, "initial_outputs.npy"), Y)
    print(f"✅ Saved processed data for Function {func_id}: {len(X)} samples")


def run_full_update():
    print("🚀 Starting full data update process...")

    # Get all weekly update folders
    week_dirs = get_sorted_weeks()
    print(f"found weekly updates: {[os.path.basename(w) for w in week_dirs]}")

    # Process each function
    for func_id in range(1, 9):  # Functions 1 to 8
        print(f"\nProcessing Function {func_id}...")

        # 1. Load Initial Data
        current_X, current_Y = load_initial_data(func_id)
        if current_X is None:
            continue

        print(f"  Loaded initial data: {len(current_X)} samples")

        # 2. Iterate through all weeks and apply updates
        for week_dir in week_dirs:
            week_name = os.path.basename(week_dir)
            inputs_path = os.path.join(week_dir, "inputs.txt")
            outputs_path = os.path.join(week_dir, "outputs.txt")

            try:
                inputs_list = parse_text_data(inputs_path)
                outputs_list = parse_text_data(outputs_path)

                # Check bounds
                if len(inputs_list) < func_id or len(outputs_list) < func_id:
                    print(
                        f"  ⚠️ Warning: {week_name} data too short for function {func_id}"
                    )
                    continue

                # Get data for this specific function (indices are 0-based, func_id is 1-based)
                new_x = inputs_list[func_id - 1]
                new_y = outputs_list[func_id - 1]

                # Reshape/Format Logic (copied from original script)
                new_x_arr = np.array(new_x).reshape(1, -1)

                # Handle scalar Y wrap
                if hasattr(new_y, "item"):
                    new_y = new_y.item()

                new_y_arr = np.array([new_y])
                if current_Y.ndim > 1:
                    new_y_arr = new_y_arr.reshape(1, -1)

                # Dimension check
                if new_x_arr.shape[1] != current_X.shape[1]:
                    print(f"  ❌ {week_name}: Dim mismatch. Skip.")
                    continue

                # Stack
                current_X = np.vstack((current_X, new_x_arr))
                if current_Y.ndim == 1:
                    current_Y = np.append(current_Y, new_y)
                else:
                    current_Y = np.vstack((current_Y, new_y_arr))

                print(f"  + Added data from {week_name}")

            except Exception as e:
                print(f"  ❌ Error processing {week_name}: {e}")

        # 3. Save to processed
        save_processed_data(func_id, current_X, current_Y)

    print("\n🎉 Data update complete!")


if __name__ == "__main__":
    run_full_update()
