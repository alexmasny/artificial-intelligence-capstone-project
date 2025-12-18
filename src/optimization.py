import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

def get_strategy(func_id):
    """
    Returns the exploration/exploitation parameter (kappa)
    and any specific flags for the function based on Week 2 Analysis.
    """
    # Default balanced strategy
    strategy = {
        'kappa': 2.576,  # 99% confidence interval
        'description': 'Balanced',
        'biased_sampling': False
    }

    # GROUP 1: Exploitation (Strong Correlations)
    if func_id in [2, 5, 6]:
        strategy['kappa'] = 1.96 # 95% CI - More greedy
        strategy['description'] = 'Exploitation (Strong Signal)'

    # GROUP 2: Exploration (Noisy/Weak Correlations)
    elif func_id in [1, 7]:
        strategy['kappa'] = 5.0 # Very high variance tolerance
        strategy['description'] = 'High Exploration (Unknown Regions)'

    # GROUP 3: Domain Knowledge / Biased Sampling
    elif func_id == 8:
        strategy['kappa'] = 1.96
        strategy['description'] = 'Exploitation + Dim Reduction'
        strategy['biased_sampling'] = True # Special flag for Func 8

    # Function 3: The conflict zone (Correlation says low, Model says mid)
    # We act on your report's decision: Trust the Model (Exploration)
    elif func_id == 3:
        strategy['kappa'] = 3.0
        strategy['description'] = 'Model Trust (Resolving Linear/Non-linear conflict)'

    return strategy

def suggest_next_point(func_id, X_train, y_train):
    strategy = get_strategy(func_id)
    print(f"Optimizing Function {func_id}: {strategy['description']}")

    # 1. Preprocessing
    # Scale targets to mean=0, std=1. Critical for Function 5 (High magnitude)
    # and Function 2 (Noisy).
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    # 2. Gaussian Process Definition
    # Matern kernel is less smooth than RBF, better for jagged real-world functions
    # WhiteKernel accounts for observation noise (sigma_n^2)
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)

    gpr = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=20,
                                   normalize_y=False) # We scaled manually

    gpr.fit(X_train, y_scaled)

    # 3. Candidate Generation (The "Search Space")
    n_dim = X_train.shape[1]
    n_samples = 50000

    # Start with uniform random candidates [0, 1]
    X_candidates = np.random.uniform(0, 1, (n_samples, n_dim))

    # --- Feature Engineering / Biased Sampling for Function 8 ---
    if strategy['biased_sampling']:
        print("   -> Applying Biased Sampling (Low X1, X3)")
        # Replace 50% of candidates with points where X1 and X3 are < 0.2
        # This reflects your EDA finding of strong negative correlation
        n_bias = int(n_samples * 0.5)

        # We force X1 and X3 (indices 0 and 2) to be small
        X_candidates[:n_bias, 0] = np.random.uniform(0, 0.15, n_bias)
        X_candidates[:n_bias, 2] = np.random.uniform(0, 0.15, n_bias)

    # 4. Acquisition Function (UCB)
    # mu + kappa * sigma
    mu, std = gpr.predict(X_candidates, return_std=True)
    ucb_scores = mu + strategy['kappa'] * std

    # 5. Select Best Point
    best_idx = np.argmax(ucb_scores)
    next_point = X_candidates[best_idx]

    return next_point
