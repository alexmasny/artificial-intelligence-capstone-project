# Artificial Intelligence Capstone Project – Black-Box Optimization (BBO)

## Project Goals and Technologies

### Project Goals

- The primary goal of this capstone project is to build and optimize a machine learning model within a simulated black-box environment. This involves:
- Applying ML techniques to solve a real-world style optimization problem where the internal workings of the functions are unknown.
- Demonstrating the ability to make iterative, evidence-based improvements to model performance without access to exact evaluation metrics during the process.
- Optimizing eight unknown functions (black-box functions) with varying dimensions (2D to 8D) using limited queries.
- Balancing exploration and exploitation strategies to find the maximum output for each function.
- Documenting the process and reflecting on the strategies used, challenges faced, and adjustments made.
- Creating a tangible, portfolio-ready artefact (this GitHub repository) that showcases the ability to tackle complex ML challenges.

### Key Technologies & Concepts

- Black-Box Optimization (BBO): The core problem setting where the objective function's analytical form is unknown.
- Bayesian Optimization: A probabilistic model-based approach for finding the global optimum of black-box functions.
- Gaussian Processes (GP): Used as a surrogate model to approximate the unknown functions and estimate uncertainty.
- Acquisition Functions: specifically Upper Confidence Bound (UCB), used to guide the search by balancing exploration (sampling uncertain areas) and exploitation (sampling high-performing areas).
- Python: The primary programming language used for implementation.

#### Libraries:

- numpy: For numerical operations and data manipulation.
- scikit-learn: For implementing Gaussian Process Regressor and other ML utilities.
- scipy: For scientific and technical computing.
- matplotlib: For visualizing data and optimization landscapes.

#### Exploratory Data Analysis (EDA)

Techniques used to understand the initial data distribution and feature importance.

## Weekly Progress

### Week 1

- Initialized project structure.
- Created virtual environment and installed dependencies (numpy, scipy, scikit-learn, matplotlib).
- Implemented load_data helper in utils.py.
- Performed Exploratory Data Analysis (EDA), including statistical summaries, visualizations, and feature importance analysis.
- Created eda_analysis.ipynb and eda_analisys.py
- Developed a Bayesian Optimization engine utilizing Gaussian Process Regression (Matern kernel) and Upper Confidence Bound (UCB) acquisition, featuring dynamic exploration-exploitation strategies and heuristic biased sampling for high-dimensional constraints.
- bayesian_optimization.ipynb

### Week 2

- Refined Bayesian Optimization strategy based on Week 1 results and feedback.
- Updated the dataset with new query results (11 data points per function).
- Implemented a tailored exploration-exploitation strategy using dynamic kappa values for the UCB acquisition function,
adjusting based on function characteristics (e.g., noise, correlation).
- Applied heuristic biased sampling for the high-dimensional Function 8 to focus on promising regions.
- Integrated StandardScaler for target variable normalization to improve GP stability, particularly for functions with large output ranges.
- Generated and submitted the second round of queries.
  - week2_bayesian_optimization.ipynb

### Week 3

- Implemented a **Hybrid SVM-GP Optimization Strategy**:
  - **SVC Filtering:** Used Support Vector Classification (RBF kernel) to classify and prune the search space for high-dimensional functions (e.g., Function 8), filtering out low-probability regions before GP evaluation.
  - **SVR Consensus:** Integrated Support Vector Regression (SVR) as a secondary validation signal for mid-dimensional functions to refine UCB scores.
- Continued applying `StandardScaler` and biased sampling based on accumulated data evidence.
- Generated 3rd round of queries focusing on interpretability and search space reduction.
  - week_3_svm_strategy.ipynb