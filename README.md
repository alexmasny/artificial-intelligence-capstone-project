# Artificial Intelligence Capstone Project – Black-Box Optimization (BBO)

## **Project Overview**

### **Documentation Links**
- **[Datasheet](datasheet.md)**: Details about the dataset, potential biases, and data generation process.
- **[Model Card](model_card.md)**: Model architecture, intended use, limitations, and ethical considerations.

### **Non-Technical Summary**
Imagine trying to find the highest peak in a dark, hilly landscape where every step you take is extremely expensive. You can only afford to take about 20 steps total. This project builds a "smart compass" (a machine learning model) that helps decide exactly where to step next. By combining multiple AI models acting together like a cautious committee, it learns the shape of the unknown landscape from very little information to avoid bad guesses. The outcome is a system that can efficiently find the optimal settings for complex scenarios—like tuning chemical reactions or edge AI devices—saving significant time and resources without needing thousands of trial-and-error physical experiments.

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

#### **Phase 7: Adaptive AutoML & Robust Acquisition (Week 7)**

* **Method:** Per-function architecture search (AutoML) and enhanced acquisition using UCB + Repulsion.
* **Strategy:**
    * **Adaptive Complexity (AutoML):** Implemented `RandomizedSearchCV` to automatically select the optimal MLP architecture (`hidden_layer_sizes`, `alpha`) for each function based on historical points. This allows the model to adapt to varying degrees of function complexity.
    * **UCB-Repulsion Acquisition:** Enhanced the acquisition function with Upper Confidence Bound (UCB) and a Gaussian Repulsion Penalty. The penalty actively "pushes" the optimizer away from already-sampled high-value points, forcing it to explore novel high-potential regions within the Trust Region.
    * **Perturbed Initialization:** Solved "zero-gradient" saddle points by initializing the gradient ascent with small random perturbations from the current best point.

#### **Phase 8: LLM-Informed Optimization (Week 8 - Current)**

* **Method:** "LLM-Critic" hybrid strategy layered on the Week 7 numerical engine.
* **Strategy:**
    * **Numerical Engine (Primary):** Reused Adaptive AutoML + Trust Region optimization to generate valid candidate submissions with precise floating-point control.
    * **Prompt Generation (Secondary):** Built Few-Shot Chain-of-Thought style prompts from historical inputs/outputs to analyze token usage, context windows, and framing effects for reflection.
    * **Separation of Concerns:** LLM prompts are used for conceptual analysis only; final submissions remain purely numerical to avoid hallucinated values.

#### **Phase 9: Scaling Laws & Emergent Behaviors (Week 9 - Current)**

* **Method:** Massive Neural Ensembling (Bagging) + Trust Regions + Repulsion.
* **Strategy:**
    * **Scaling Laws (Ensemble Size):** Scaled the surrogate model from 3 to **20 Neural Networks**. Following scaling laws, this linearly reduced the variance of uncertainty estimates, providing a robust signal for the UCB acquisition function without needing more data.
    * **Emergent Robustness:** Leveraged the "wisdom of the crowd" to handle high-dimensional edge cases (Func 7 & 8). The large ensemble naturally smoothed out local noise and prevented boundary saturation, an emergent behavior not explicitly programmed but resulting from scale.
    * **Diminishing Returns:** Balanced the compute cost of 20x inference against the marginal gains in lower dimensions, accepting higher latency for the sake of stability in critical high-dim functions.

#### **Phase 10: Transparent Optimization (Week 10 - Current)**

* **Method:** Transparent Scaled Ensembles + Interpretability Logging.
* **Strategy:**
    * **Interpretability Metrics:** Transitioned from a "black-box" surrogate to a transparent one by explicitly logging internal decision metrics (Predicted Mean, Predictive Standard Deviation, Acquisition Value).
    * **Reproducibility:** Strictly fixed random seeds and logged hyperparameters to ensure every query decision is auditable and reproducible.
    * **Decision Transparency:** Explicitly tracked the trade-off between the exploitation signal (mean) and exploration signal (uncertainty) for each submission, making the optimization logic "glass-box".

#### **Phase 11: Clustering & Boundary Tightening (Week 11)**

* **Method:** Cluster-Centric Trust Regions.
* **Strategy:**
    * **High-Yield Clustering:** Selected the top subset (top 15-20%) of known data points to compute a performance-weighted "centroid," anchoring the optimization search.
    * **Dynamic Boundary Tightening:** Calculated the standard deviation (spread) within the cluster to dynamically define the Trust Region boundaries. Dense clusters enforced tight exploitation, while scattered clusters allowed broader exploration.
    * **Intra-Cluster Repulsion:** Forced purely localized exploration *between* the historically strong points of the cluster, driving the optimizer to search for unmapped peaks among known successes.

#### **Phase 12: Dimensionality Reduction & PCA (Week 12)**

* **Method:** PCA-Guided Subspace Optimization.
* **Strategy:**
    * **Variance-Based Subspaces:** Executed Principal Component Analysis (PCA) on the top-performing cluster to analyze the explained variance ratio across feature dimensions.
    * **Asymmetric Bounds:** Severely restricted search boundaries on "redundant" (low-variance) converged dimensions, while relaxing the bounds on "principal" (high-variance) dimensions.
    * **Efficient Exploration:** Maximized the value of the final queries by restricting the optimization payload strictly to active sub-domains, ignoring completely flat operational areas.

#### **Phase 13: Reinforcement Learning & Pure Exploitation (Week 13)**

* **Method:** Pure Exploitation Q-Value Maximization (Greedy Strategy).
* **Strategy:**
    * **Zero Exploration ($\\kappa = 0$):** Following RL schedules (e.g., $\\epsilon$-greedy decay), transitioned entirely from checking uncertain regions to pure exploitation (Greedy) for the final round.
    * **Temporal Difference & Q-Values:** Used the ensemble surrogate (GP + MLP) as a Q-Value estimator, where comparing historical observations against predictions implicitly models TD errors.
    * **Micro Trust Regions:** Confined gradient ascent to an extremely tight radius ($\\pm 5\\%$) around the historically absolute best known point, ensuring we climb the immediate peak.
    * **Autonomous Learning:** Paralleled AlphaGo Zero's self-play by executing model-based planning—simulating candidate points over the surrogate "mental models" before picking the optimal final action across the real black-box environmental states.

### **Summary of Progress**

| Week   | Strategy                  | Key Technologies                     | Outcome                                                                                       |
| :----- | :------------------------ | :----------------------------------- | :-------------------------------------------------------------------------------------------- |
| **1**  | Random Search             | numpy, EDA, Random Forest            | Established baseline, identified key features in 8D function.                                 |
| **2**  | Bayesian Optimization     | sklearn GP, UCB, StandardScaler      | Transitioned to model-based search; tailored exploration ($\\kappa$) to function noise.       |
| **3**  | Hybrid SVM-GP             | SVC Filtering, SVR Ensemble          | Reduced search space for High-Dim functions; improved robustness against local optima.        |
| **4**  | Neural Gradient Ascent    | MLPRegressor, L-BFGS, scipy.optimize | Used backpropagation to steer queries; captured non-linearities in complex functions.         |
| **5**  | Neural Ensembles          | NN Bagging, Ensemble Learning        | Improved robustness against overfitting; applied hierarchical feature learning concepts.      |
| **6**  | Trust Region Optimization | Localized Gradient Ascent            | Improved performance against global optima.                                                   |
| **7**  | Adaptive AutoML           | RandomizedSearchCV, UCB, Repulsion   | Optimized architecture per function; forced exploration near peaks with repulsive gradients.  |
| **8**  | LLM-Informed Optimization | LLM prompt analysis, Adaptive AutoML | Used LLMs for reflection on sequence framing; kept numerical engine for valid submissions.    |
| **9**  | Scaling Ensembles         | BaggingRegressor (N=20), TrustRegion | Reduced variance via massive ensembling; achieved emergent robustness in high dimensions.     |
| **10** | Transparent Optimization  | Decision Logging, Scaled Ensembles   | Improved interpretability of the surrogate model; reproducible and auditable query decisions. |
| **11** | Clustered Trust Regions   | Top-$K$ Centroids, Dynamic Bounds    | Tightened search limits by scaling radius based on cluster variance density.                  |
| **12** | PCA Subspace Optimization | PCA, Explained Variance              | Focused exploration purely on principal dimensions, clamping boundaries on converged inputs.  |
| **13** | RL & Pure Exploitation    | Q-Value Maximization, Micro bounds   | Executed final $\\epsilon$-greedy phase ($\\kappa=0$) around tight trust regions for max reward.|

## References

- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *NeurIPS*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
- Eriksson, D., et al. (2019). Scalable Global Optimization with Trust Regions (TuRBO). *NeurIPS*.
- Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv*.
- Silver, D., et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero). *Nature*.
