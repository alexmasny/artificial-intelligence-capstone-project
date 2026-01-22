# Artificial Intelligence Capstone Project – Black-Box Optimization (BBO)

## **Project Overview**

### **Black-Box Optimization (BBO) Capstone**

This repository documents my solution for the Black-Box Optimization challenge, simulating a real-world machine learning scenario where the objective functions are unknown, high-dimensional, and expensive to evaluate. The goal is to maximize the output of eight distinct functions (ranging from 2D to 8D) using a limited budget of weekly queries.

**Relevance:** In industries like drug discovery, chemical manufacturing, or automated hyperparameter tuning, we often treat systems as "black boxes" — we can observe inputs and outputs, but not the internal physics. This project mirrors that constraint, requiring data-driven decision-making under uncertainty.

**Career Value:** Mastering BBO equips me with the skills to handle "cold start" problems and optimize complex systems efficiently, moving beyond standard supervised learning into active learning and decision science.

### **Inputs and Outputs**

The interaction with the "black box" is structured as follows:

* **Inputs:** A query vector $X$ of dimension D (where $D \in \{2, ..., 8\}$).
  * *Constraints:* All values must be within the hypercube $[0, 1]$.
  * *Format:* `X1 - X2 - ... - Xn` (e.g., `0.123456 - 0.987654` for a 2D function).
* **Outputs:** A single continuous scalar value $Y$.
  * *Goal:* Maximize $Y$.
  * *Feedback:* After each weekly submission, one new pair ($X_{new}$, $Y_{new}$) is added to the training set.



### **Challenge Objectives**

The primary objective is not just finding the global maximum, but demonstrating a robust **optimization strategy**.

1. **Maximize Function Output:** Systematically navigate the search space to find peak values.
2. **Efficient Querying:** Use limited samples (starting with 10 points) to learn complex landscapes.
3. **Manage Trade-offs:** Balance **Exploration** (gathering information in unknown regions) vs. **Exploitation** (refining solutions in promising areas).
4. **Handle High Dimensions:** Mitigate the "curse of dimensionality" in 8D functions where data is extremely sparse.

### **Technical Approach (Living Document)**

My approach evolves iteratively as the dataset grows from 10 to ~22 points.

#### **Phase 1: Baseline & EDA (Week 1)**

* **Method:** Random Search and Random Forest-based Exploratory Data Analysis (EDA).
* **Goal:** Understand the landscape. I analyzed Pearson correlations and Feature Importance to classify functions into "simple" (linear correlations) vs. "complex" (noisy/non-linear).
* **Key Insight:** Discovered that Function 8 (8D) effectively operates on a lower-dimensional manifold (dominated by dimensions 1 and 3).

#### **Phase 2: Bayesian Optimization (Week 2)**

* **Method:** Gaussian Process (GP) Regression with a Matern kernel + Upper Confidence Bound (UCB) acquisition function.
* **Strategy:**
  * **Dynamic Kappa:** Tuned the exploration parameter ($\kappa$) per function. High $\kappa$ (5.0) for noisy functions (1 & 7) to encourage exploration; low $\kappa$ (1.96) for "well-behaved" functions.
  * **Data Transformation:** Applied `StandardScaler` to targets (e.g., Function 5) to stabilize GP variance estimation in high-magnitude ranges.
  * **Heuristic Biased Sampling:** For Function 8, I biased the candidate generation toward low values for dimensions $X_1$ and $X_3$ based on strong negative correlations found in EDA, while allowing other dimensions to vary freely.

#### **Phase 3: Hybrid SVM-GP Strategy (Week 3 - Current)**

* **Method:** Integration of Support Vector Machines (SVM) to assist the Gaussian Process.
* **Refinement:**
  * **SVC Pruning:** For high-dimensional functions (6D-8D), I implemented a Support Vector Classifier (RBF kernel) to pre-filter the search space, discarding "low-yield" regions before the GP evaluation. This reduces the search volume for the optimizer.
  * **SVR Consensus:** For mid-dimensional functions, an SVR model acts as a secondary validator. Queries are prioritized only if both the GP (probabilistic) and SVR (deterministic) agree on their potential.

# Artificial Intelligence Capstone Project – Black-Box Optimization (BBO)

## **Project Overview**

### **Black-Box Optimization (BBO) Capstone**

This repository documents my solution for the Black-Box Optimization challenge, simulating a real-world machine learning scenario where the objective functions are unknown, high-dimensional, and expensive to evaluate. The goal is to maximize the output of eight distinct functions (ranging from 2D to 8D) using a limited budget of weekly queries.

**Relevance:** In industries like drug discovery, chemical manufacturing, or automated hyperparameter tuning, we often treat systems as "black boxes" — we can observe inputs and outputs, but not the internal physics. This project mirrors that constraint, requiring data-driven decision-making under uncertainty.

**Career Value:** Mastering BBO equips me with the skills to handle "cold start" problems and optimize complex systems efficiently, moving beyond standard supervised learning into active learning and decision science.

### **Inputs and Outputs**

The interaction with the "black box" is structured as follows:

* **Inputs:** A query vector $X$ of dimension D (where $D \in \{2, ..., 8\}$).
  * *Constraints:* All values must be within the hypercube $[0, 1]$.
  * *Format:* `X1 - X2 - ... - Xn` (e.g., `0.123456 - 0.987654` for a 2D function).
* **Outputs:** A single continuous scalar value $Y$.
  * *Goal:* Maximize $Y$.
  * *Feedback:* After each weekly submission, one new pair ($X_{new}$, $Y_{new}$) is added to the training set.



### **Challenge Objectives**

The primary objective is not just finding the global maximum, but demonstrating a robust **optimization strategy**.

1. **Maximize Function Output:** Systematically navigate the search space to find peak values.
2. **Efficient Querying:** Use limited samples (starting with 10 points) to learn complex landscapes.
3. **Manage Trade-offs:** Balance **Exploration** (gathering information in unknown regions) vs. **Exploitation** (refining solutions in promising areas).
4. **Handle High Dimensions:** Mitigate the "curse of dimensionality" in 8D functions where data is extremely sparse.

### **Technical Approach (Living Document)**

My approach evolves iteratively as the dataset grows from 10 to ~22 points.

#### **Phase 1: Baseline & EDA (Week 1)**

* **Method:** Random Search and Random Forest-based Exploratory Data Analysis (EDA).
* **Goal:** Understand the landscape. I analyzed Pearson correlations and Feature Importance to classify functions into "simple" (linear correlations) vs. "complex" (noisy/non-linear).
* **Key Insight:** Discovered that Function 8 (8D) effectively operates on a lower-dimensional manifold (dominated by dimensions 1 and 3).

#### **Phase 2: Bayesian Optimization (Week 2)**

* **Method:** Gaussian Process (GP) Regression with a Matern kernel + Upper Confidence Bound (UCB) acquisition function.
* **Strategy:**
  * **Dynamic Kappa:** Tuned the exploration parameter ($\kappa$) per function. High $\kappa$ (5.0) for noisy functions (1 & 7) to encourage exploration; low $\kappa$ (1.96) for "well-behaved" functions.
  * **Data Transformation:** Applied `StandardScaler` to targets (e.g., Function 5) to stabilize GP variance estimation in high-magnitude ranges.
  * **Heuristic Biased Sampling:** For Function 8, I biased the candidate generation toward low values for dimensions $X_1$ and $X_3$ based on strong negative correlations found in EDA, while allowing other dimensions to vary freely.

#### **Phase 3: Hybrid SVM-GP Strategy (Week 3 - Current)**

* **Method:** Integration of Support Vector Machines (SVM) to assist the Gaussian Process.
* **Refinement:**
  * **SVC Pruning:** For high-dimensional functions (6D-8D), I implemented a Support Vector Classifier (RBF kernel) to pre-filter the search space, discarding "low-yield" regions before the GP evaluation. This reduces the search volume for the optimizer.
  * **SVR Consensus:** For mid-dimensional functions, an SVR model acts as a secondary validator. Queries are prioritized only if both the GP (probabilistic) and SVR (deterministic) agree on their potential.


#### **Phase 4: Neural Surrogate & Gradient Ascent (Week 4\)**

* **Method:** Neural Network (MLPRegressor) acting as a differentiable surrogate model, optimized via Gradient Ascent (L-BFGS-B).  
* **Strategy:**  
  * **Gradient Steering:** Shifted from sampling to "steering". Used tanh activations to create smooth, differentiable decision surfaces, allowing calculation of gradients ($\\nabla f(x)$) to guide queries toward theoretical maxima.  
  * **Ensembling:** Combined the Neural Network (high flexibility) with the Gaussian Process (uncertainty management) to prevent overfitting on the small dataset (13 points).  
  * **Solver Choice:** Utilized L-BFGS instead of Adam for superior convergence stability on small-batch data.

#### **Phase 5: Neural Ensembles & Deep Learning Concepts (Week 5\)**

* **Method:** Bagging Ensembles of Multi-Layer Perceptrons (MLP) to mimic Deep Learning architectural robustness.  
* **Strategy:**  
  * **Neural Ensembles:** Trained multiple independent NNs with different random seeds and averaged their predictions. This mitigates the variance and "hallucinations" of single networks trained on sparse data (14 points).  
  * **Hierarchical Features:** Slightly deepened the network architecture to capture non-linear feature interactions (inspired by Deep Learning feature hierarchies), while relying on the ensemble for regularization.  
  * **Robust Optimization:** Performed gradient ascent on the *averaged* ensemble surface, ensuring the chosen query satisfies the consensus of multiple models.

#### **Phase 6: Trust Region Optimization (Week 6)**

* **Method:** Localized Gradient Ascent within a dynamic "Trust Region".
* **Strategy:**
    * **Trust Regions (Receptive Fields):** Constrained the optimization search space to a hypercube ($\pm 20\%$) around the current best known point. This prevents the optimizer from exploiting global linear trends (saturating at 1.0) and forces it to refine local optima, analogous to CNN receptive fields focusing on local features.
    * **Ensemble Pooling:** Continued using Neural Bagging to smooth out the decision surface, acting as an "Average Pooling" layer to filter noise from individual surrogate models.

### **Summary of Progress**

| Week | Strategy | Key Technologies | Outcome |
| :---- | :---- | :---- | :---- |
| **1** | Random Search | numpy, EDA, Random Forest | Established baseline, identified key features in 8D function. |
| **2** | Bayesian Optimization | sklearn GP, UCB, StandardScaler | Transitioned to model-based search; tailored exploration ($\\kappa$) to function noise. |
| **3** | Hybrid SVM-GP | SVC Filtering, SVR Ensemble | Reduced search space for High-Dim functions; improved robustness against local optima. |
| **4** | Neural Gradient Ascent | MLPRegressor, L-BFGS, scipy.optimize | Used backpropagation to steer queries; captured non-linearities in complex functions. |
| **5** | Neural Ensembles | NN Bagging, Ensemble Learning | Improved robustness against overfitting; applied hierarchical feature learning concepts. |
| **6** | Trust Region Optimization | Localized Gradient Ascent | Improved performance against global optima. |
