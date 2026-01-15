import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_data
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def perform_eda():
    # Create plots directory
    if not os.path.exists("eda_plots"):
        os.makedirs("eda_plots")

    report_content = "# EDA Summary Report\n\n"

    for function_id in range(1, 9):
        print(f"Analyzing Function {function_id}...")
        inputs, outputs = load_data(function_id)

        if inputs is None:
            print(f"Skipping Function {function_id} (Data not found)")
            continue

        # 1. Data Structure
        n_samples, n_features = inputs.shape
        report_content += f"## Function {function_id}\n"
        report_content += f"- **Dimensionality:** {n_features}D\n"
        report_content += f"- **Samples:** {n_samples}\n"

        # Check for missing values
        missing_inputs = np.isnan(inputs).sum()
        missing_outputs = np.isnan(outputs).sum()
        if missing_inputs > 0 or missing_outputs > 0:
            report_content += f"- **WARNING:** Found missing values! Inputs: {missing_inputs}, Outputs: {missing_outputs}\n"

        # 2. Descriptive Statistics
        report_content += "### Statistics\n"
        report_content += "| Feature | Mean | Std | Min | Max |\n"
        report_content += "|---|---|---|---|---|\n"

        for i in range(n_features):
            feat = inputs[:, i]
            report_content += f"| X{i+1} | {np.mean(feat):.4f} | {np.std(feat):.4f} | {np.min(feat):.4f} | {np.max(feat):.4f} |\n"

            # Check for zero variance
            if np.std(feat) == 0:
                report_content += f"- **WARNING:** Feature X{i+1} has zero variance.\n"

        report_content += f"| Target (Y) | {np.mean(outputs):.4f} | {np.std(outputs):.4f} | {np.min(outputs):.4f} | {np.max(outputs):.4f} |\n\n"

        # 3. Visual Exploration
        # Histograms
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(outputs, bins=10, color="skyblue", edgecolor="black")
        plt.title(f"Function {function_id}: Target Distribution")
        plt.xlabel("Y")
        plt.ylabel("Frequency")

        # Boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(outputs)
        plt.title(f"Function {function_id}: Target Boxplot")

        plt.tight_layout()
        plt.savefig(f"eda_plots/function_{function_id}_dist.png")
        plt.close()

        # Scatter Plots
        if n_features == 2:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                inputs[:, 0], inputs[:, 1], outputs, c=outputs, cmap="viridis"
            )
            plt.colorbar(sc, label="Y")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Y")
            plt.title(f"Function {function_id}: 3D Surface")
            plt.savefig(f"eda_plots/function_{function_id}_scatter.png")
            plt.close()
        else:
            # For > 2D, plot each feature vs Y
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            plt.figure(figsize=(15, 4 * n_rows))
            for i in range(n_features):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.scatter(inputs[:, i], outputs, alpha=0.7)
                plt.xlabel(f"X{i+1}")
                plt.ylabel("Y")
                plt.title(f"X{i+1} vs Y")
            plt.tight_layout()
            plt.savefig(f"eda_plots/function_{function_id}_scatter.png")
            plt.close()

        # 4. Relationships & Importance
        # Correlation
        report_content += "### Correlations (Pearson)\n"
        # Combine inputs and outputs for correlation matrix
        data_combined = np.hstack((inputs, outputs.reshape(-1, 1)))
        corr_matrix = np.corrcoef(data_combined, rowvar=False)
        target_corr = corr_matrix[:-1, -1]  # Correlation of inputs with target

        for i, corr in enumerate(target_corr):
            report_content += f"- **X{i+1} vs Y:** {corr:.4f}"
            if abs(corr) > 0.5:
                report_content += " **(Strong)**"
            report_content += "\n"

        report_content += "\n"

        # Feature Importance (Random Forest)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(inputs, outputs.ravel())
        importances = rf.feature_importances_

        report_content += "### Feature Importance (Random Forest)\n"
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            report_content += f"- **X{idx+1}:** {importances[idx]:.4f}\n"

        report_content += "\n---\n\n"

    with open("EDA_SUMMARY.md", "w") as f:
        f.write(report_content)

    print("EDA Analysis Complete. Report saved to EDA_SUMMARY.md")


if __name__ == "__main__":
    perform_eda()
