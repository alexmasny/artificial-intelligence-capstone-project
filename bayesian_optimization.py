import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from utils import load_data
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def generate_next_point(func_id, X_train, y_train):
    # 1. Standardization
    # Always scale the target variable
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # 2. Model Setup
    # Matern kernel is generally robust for BO
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr.fit(X_train, y_train_scaled)

    # 3. Kappa Selection (Acquisition Strategy)
    if func_id in [2, 3, 5, 8]:
        kappa = 1.96 # Exploitation
    elif func_id in [1, 7]:
        kappa = 5.0 # Exploration
    else: # 4, 6
        kappa = 2.57 # Balanced

    # 4. Candidate Generation
    n_candidates = 10000
    n_dims = X_train.shape[1]
    
    # Standard random sampling
    X_candidates = np.random.uniform(0, 1, (n_candidates, n_dims))

    # Special handling for Function 8 (Biased Sampling)
    if func_id == 8:
        # X1 and X3 are strong negative correlations -> bias towards 0.0 - 0.5
        # We'll replace 70% of candidates with biased samples for X1 and X3
        n_biased = int(0.7 * n_candidates)
        
        # Generate biased components
        biased_X1 = np.random.uniform(0, 0.5, n_biased)
        biased_X3 = np.random.uniform(0, 0.5, n_biased)
        
        # Inject into candidates
        X_candidates[:n_biased, 0] = biased_X1 # X1 is index 0
        X_candidates[:n_biased, 2] = biased_X3 # X3 is index 2
        
        # The rest of the dimensions for these biased candidates are still uniform 0-1 
        # (already set by the initial uniform call)

    # 5. Acquisition Function (UCB)
    mean, std = gpr.predict(X_candidates, return_std=True)
    ucb = mean + kappa * std
    
    # Select best candidate
    best_idx = np.argmax(ucb)
    next_point = X_candidates[best_idx]

    return next_point

def main():
    print("Starting Bayesian Optimization...")
    print("-" * 30)

    new_points = {}

    for func_id in range(1, 9):
        inputs, outputs = load_data(func_id)
        
        if inputs is None:
            print(f"Function {func_id}: Data not found.")
            continue

        next_point = generate_next_point(func_id, inputs, outputs)
        new_points[func_id] = next_point
        
        # Format point for display
        point_str = ", ".join([f"{x:.6f}" for x in next_point])
        print(f"Function {func_id} Next Point: [{point_str}]")

        # Verification Warnings
        if func_id == 2:
            # X1 should be high (strong positive correlation)
            if next_point[0] < 0.5:
                print(f"  WARNING: Function 2 X1 is {next_point[0]:.4f} (Expected high value > 0.5)")
        
        if func_id == 3:
            # X3 should be low (strong negative correlation)
            if next_point[2] > 0.5:
                print(f"  WARNING: Function 3 X3 is {next_point[2]:.4f} (Expected low value < 0.5)")

    print("-" * 30)
    print("Optimization Complete.")

if __name__ == "__main__":
    main()
