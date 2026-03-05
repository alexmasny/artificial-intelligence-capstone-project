## Model Card: Scaled Hybrid Surrogate Optimizer (SHSO)

## 1. Overview

- Name: Scaled Hybrid Surrogate Optimizer (SHSO)
- Type: Ensemble Regression Model with Trust Region Acquisition.
- Version: 1.0 (Week 10 Iteration)
- Core Architecture: Bagging of 20 Tuned Multi-Layer Perceptrons (MLPs) paired with a Gaussian Process (GP) using a Matern kernel.

## 2. Intended Use

- Suitable Tasks: Optimizing continuous, expensive-to-evaluate black-box functions (e.g., hyperparameter tuning, chemical yield optimization) where the query budget is strictly limited ($\le 20$ evaluations).
- Cases to Avoid: Highly discontinuous step-functions, tasks requiring processing of millions of data points (where deep learning frameworks like PyTorch would be more efficient), and functions with sharp, needle-like Rastrigin peaks.

## 3. Details (Strategy Evolution over 10 Rounds)

The optimization strategy evolved significantly from basic heuristics to a complex automated pipeline:

- Rounds 1-3: Began with Random Search and basic Gaussian Process (GP) with Upper Confidence Bound (UCB). Introduced SVM to prune low-yield spaces.
- Rounds 4-5: Shifted to Neural Surrogates (MLPRegressor) and Gradient Ascent to capture non-linearities, using bagging (ensembles) to prevent overfitting.
- Rounds 6-7: Encountered "boundary saturation" (getting stuck at 1.0). Implemented Trust Regions ($\pm 20\%$ receptive fields) and Adaptive AutoML (tuning architecture per function).
- Rounds 8-10: Scaled the ensemble to 20 models for robust uncertainty estimation. Integrated explicit Repulsion Penalties and transparent decision logging (Mean vs. Std) to evaluate exploration vs. exploitation mathematically.

## 4. Performance

- Summary: The model successfully navigated away from local boundary saturation in mid-dimensional functions and consistently found high-yield regions.
- Metrics: Performance was tracked not just by raw $Y$ maximization, but by "Sample Efficiency" (rate of improvement per query) and internal UCB metrics (Predicted Mean, Ensemble Standard Deviation).

## 5. Assumptions and Limitations

- Assumptions: The model assumes Smoothness and Continuity in the underlying function due to the use of tanh activations and the Matern kernel.
- Limitations:
1. Curse of Dimensionality: In 6D and 8D, the model's high confidence is often an illusion caused by sparse data.
2. Edge-Sampling Bias: The optimizer tends to cluster queries around boundaries (e.g., repeatedly suggesting 0.000 for inputs) and initially discovered peaks, leaving "dark zones" in the hypercube interior unexplored.

## 6. Ethical Considerations and Transparency

- Transparency: The model features a "Decision Log" that explicitly prints the Predicted Mean, Uncertainty, and Penalty weights for every query. This allows any auditor or collaborator to understand exactly why the model chose a specific data point, ensuring reproducibility.
- Real-World Adaptation: By acknowledging its own sampling biases and hardware/data constraints (Edge AI parallels), the model avoids "overpromising" global optima in sparse high-dimensional scenarios, which is crucial for responsible ML deployment in industries like drug discovery or manufacturing.
