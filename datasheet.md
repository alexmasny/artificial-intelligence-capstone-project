# Datasheet: Black-Box Optimization (BBO) Capstone Dataset

## 1. Motivation

Why was this dataset created?
This dataset was created as part of the Artificial Intelligence Capstone Project to simulate real-world Black-Box Optimization (BBO) tasks. It supports the training and evaluation of surrogate machine learning models designed to find the global maxima of expensive-to-evaluate, unknown functions.

## 2. Composition

What does it contain?
The dataset consists of 8 distinct subsets, corresponding to 8 continuous mathematical functions of increasing dimensionality (from 2D to 8D).

Size: As of Round 10, each subset contains exactly 19 data points (pairs of inputs and outputs).

Format: Features ($X$) are numerical vectors bounded within a hypercube $[0.0, 1.0]^D$. Targets ($Y$) are continuous scalar values ($\mathbb{R}$). Stored as standard NumPy arrays (.npy).

Gaps & Sparsity: There are no missing values. However, there is severe data sparsity in the high-dimensional sets (Function 7 and 8), where 19 points are mathematically insufficient to cover the volume of the search space.

## 3. Collection Process

How were the queries generated?

Initial Baseline: The first 10 points for each function were provided as a static starting baseline.

Iterative Sampling: The subsequent 9 points were collected sequentially (one per week) over a 9-week timeframe.

Strategy: Queries were not generated randomly. They were actively selected using evolving Bayesian Optimization strategies (Gaussian Processes, SVM pruning, Adaptive Neural Ensembles) to balance exploration and exploitation. This creates a strong "exploitation bias" where points are clustered around promising local optima.

## 4. Preprocessing and Uses

Have you applied any transformations?
Yes. Before being fed into the surrogate models, both the inputs ($X$) and targets ($Y$) are strictly normalized using sklearn.preprocessing.StandardScaler to have zero mean and unit variance. This stabilizes the neural network gradients and Gaussian Process covariance calculations.
Intended Uses: Training surrogate models to predict function topographies.
Inappropriate Uses: The dataset is highly biased towards specific local regions discovered during optimization. It should not be used to train generalized regression models that require a uniform understanding of the entire $[0, 1]^D$ domain.

## 5. Distribution and Maintenance

Where is the data available?
The data is hosted locally and within this public GitHub repository in the data/ directory. It is maintained by the repository owner for the duration of the Capstone project.