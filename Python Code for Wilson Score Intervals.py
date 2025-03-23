import numpy as np
from statsmodels.stats.proportion import proportion_confint

# Test set size (assuming 20% of 416 samples)
n = 84

# Model metrics from your table
model_data = {
    "Logistic Regression": {"Accuracy": 0.73, "Precision": 0.76, "Recall": 0.74, "F1 Score": 0.75, "CV Score": 0.64},
    "Decision Tree": {"Accuracy": 0.73, "Precision": 0.74, "Recall": 0.76, "F1 Score": 0.75, "CV Score": 0.67},
    "Random Forest": {"Accuracy": 0.68, "Precision": 0.69, "Recall": 0.76, "F1 Score": 0.72, "CV Score": 0.67},
    "Gradient Boosting": {"Accuracy": 0.69, "Precision": 0.70, "Recall": 0.76, "F1 Score": 0.73, "CV Score": 0.67},
    "SVM": {"Accuracy": 0.71, "Precision": 0.68, "Recall": 0.89, "F1 Score": 0.77, "CV Score": 0.67},
    "K-Nearest Neighbors": {"Accuracy": 0.61, "Precision": 0.63, "Recall": 0.71, "F1 Score": 0.67, "CV Score": 0.50}
}

# Calculate Wilson score intervals
results_with_ci = {}
for model, metrics in model_data.items():
    results_with_ci[model] = {}
    for metric, value in metrics.items():
        if metric != "CV Score":  # Only for classification metrics
            # Wilson score interval calculation
            successes = round(value * n)
            lower, upper = proportion_confint(successes, n, alpha=0.05, method='wilson')
            results_with_ci[model][metric] = (value, lower, upper)
        else:
            # For CV scores, use a standard approximation based on CV fold count
            std_dev = 0.02  # Approximation based on 5-fold CV
            lower = max(0, value - 1.96 * std_dev)
            upper = min(1, value + 1.96 * std_dev)
            results_with_ci[model][metric] = (value, lower, upper)

# Print formatted results
print("\nModel Performance with 95% Wilson Score Confidence Intervals:\n")
print("{:<25} {:<12} {:<25}".format("Model", "Metric", "Value (95% CI)"))
print("-" * 60)
for model, metrics in results_with_ci.items():
    for metric, (value, ci_lower, ci_upper) in metrics.items():
        print(f"{model:<25} {metric:<12} {value:.2f} ({ci_lower:.2f} - {ci_upper:.2f})")