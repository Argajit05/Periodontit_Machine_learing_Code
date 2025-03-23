#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplementary File 2: Periodontitis Analysis

This code performs comprehensive analysis of dental clinic data to investigate 
the relationship between periodontitis and systemic health indicators.

Original data file: Dental_Data_18.02.25.xlsx
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Set professional style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------------------------------------------------------
# 1. DATA LOADING AND EXPLORATION
# -----------------------------------------------------------------------------

# Load the dataset
file_path = "Dental_Data_18.02.25.xlsx"  # Update with your file path
xls = pd.ExcelFile(file_path)
print(f"Available sheets: {xls.sheet_names}")

# Load data from the first sheet
df = pd.read_excel(xls, sheet_name="Sheet1")

# Get dataset dimensions and key variables
dimensions = df.shape
key_variables = df.columns.tolist()
print(f"Dataset dimensions: {dimensions[0]} rows and {dimensions[1]} columns")
print(f"Key variables: {key_variables}")

# -----------------------------------------------------------------------------
# 2. MISSING DATA ANALYSIS
# -----------------------------------------------------------------------------

# Check for missing values
missing_values = df.isnull().sum()
missing_columns = missing_values[missing_values > 0]
print("\nColumns with missing values:")
print(missing_columns)

# -----------------------------------------------------------------------------
# 3. DATA CLEANING
# -----------------------------------------------------------------------------

# Create a copy of the dataset for cleaning
df_cleaned = df.copy()

# Handling outliers in numerical columns
numerical_cols = ['Blood Sugar', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'CPI Score']
print("\nHandling outliers in numerical variables using IQR method...")

for col in numerical_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with median
    median_value = df_cleaned[col].median()
    df_cleaned[col] = np.where((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound), 
                               median_value, df_cleaned[col])

# Standardizing 'Sex' column
df_cleaned['Sex'] = df_cleaned['Sex'].str.strip().str.capitalize()

# Converting categorical variables to binary format (0 = No, 1 = Yes)
categorical_cols = ['Dental Visits Last Year (Yes/No)', 'Mobile Teeth (Yes/No)', 
                    'Floss Usage (Yes/No)', 'Cardiovascular Disease (Yes/No)', 'Periodontitis']
yes_no_mapping = {'Yes': 1, 'No': 0}

for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].map(yes_no_mapping)

# Convert categorical variables with multiple unique values using Label Encoding
label_cols = ['Oral Habits', 'Oral Findings', 'Any other systemic Diseases', 
              'Oral Diseases', 'History of Medication in last 6 months']
label_encoder = LabelEncoder()

for col in label_cols:
    df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col].astype(str))

print("\nData cleaning completed.")

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

print("\nCreating engineered features...")

# 1. Oral Health Behavior Index (OHBI)
df_cleaned['Oral Health Behavior Index'] = df_cleaned[['Floss Usage (Yes/No)',
                                                      'Dental Visits Last Year (Yes/No)',
                                                      'Mobile Teeth (Yes/No)']].sum(axis=1)

# 2. Blood Pressure Categories
def categorize_bp(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated"
    elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        return "Hypertension Stage 1"
    elif (140 <= systolic <= 180) or (90 <= diastolic <= 120):
        return "Hypertension Stage 2"
    else:
        return "Hypertensive Crisis"

df_cleaned['Blood Pressure Category'] = df_cleaned.apply(
    lambda row: categorize_bp(row['Systolic Blood Pressure'], row['Diastolic Blood Pressure']), 
    axis=1)

# 3. Age Group Classification
df_cleaned['Age Group'] = pd.cut(df_cleaned['Age'], 
                                bins=[0, 18, 35, 50, 65, 100],
                                labels=['Child', 'Young Adult', 'Middle-Aged', 'Senior', 'Elderly'])

# 4. Systemic Health Risk Score
df_cleaned['Systemic Health Risk Score'] = df_cleaned[['Cardiovascular Disease (Yes/No)',
                                                     'Hypertension']].applymap(
    lambda x: 1 if x == 'Yes' else 0).sum(axis=1) + (df_cleaned['Any other systemic Diseases'] > 0).astype(int)

# 5. Oral Disease Severity Score
df_cleaned['Oral Disease Severity'] = df_cleaned['CPI Score'] + df_cleaned['Periodontitis']

# Save final enhanced dataset
enhanced_dataset = df_cleaned.copy()
print("Feature engineering completed.")

# -----------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------

print("\nPerforming exploratory data analysis...")

# Function to save figures with consistent formatting
def save_figure(filename, dpi=300):
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

# 1. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(enhanced_dataset['Age'], bins=20, kde=True)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Age Distribution', fontsize=14)
save_figure('age_distribution.png')

# 2. Community Periodontal Index (CPI) Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(enhanced_dataset['CPI Score'], bins=10, kde=True)
plt.xlabel('CPI Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('CPI Score Distribution', fontsize=14)
save_figure('cpi_score_distribution.png')

# 3. Blood Pressure Distributions
plt.figure(figsize=(8, 5))
sns.histplot(enhanced_dataset['Systolic Blood Pressure'], bins=15, kde=True, color='blue', label="Systolic")
sns.histplot(enhanced_dataset['Diastolic Blood Pressure'], bins=15, kde=True, color='red', label="Diastolic", alpha=0.6)
plt.xlabel('Blood Pressure (mmHg)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Blood Pressure Distribution', fontsize=14)
plt.legend()
save_figure('blood_pressure_distribution.png')

# 4. Blood Sugar Levels Distribution
plt.figure(figsize=(8, 5))
sns.histplot(enhanced_dataset['Blood Sugar'], bins=20, kde=True, color='purple')
plt.xlabel('Blood Sugar Level (mg/dL)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Blood Sugar Level Distribution', fontsize=14)
save_figure('blood_sugar_distribution.png')

# Box plots to identify outliers
# 5. Box Plot for Age
plt.figure(figsize=(6, 5))
sns.boxplot(y=enhanced_dataset['Age'], color='orange')
plt.ylabel('Age', fontsize=12)
plt.title('Box Plot of Age', fontsize=14)
save_figure('age_boxplot.png')

# 6. Box Plot for Blood Sugar Levels
plt.figure(figsize=(6, 5))
sns.boxplot(y=enhanced_dataset['Blood Sugar'], color='purple')
plt.ylabel('Blood Sugar (mg/dL)', fontsize=12)
plt.title('Box Plot of Blood Sugar Levels', fontsize=14)
save_figure('blood_sugar_boxplot.png')

# 7. Box Plot for CPI Score
plt.figure(figsize=(6, 5))
sns.boxplot(y=enhanced_dataset['CPI Score'], color='green')
plt.ylabel('CPI Score', fontsize=12)
plt.title('Box Plot of CPI Score', fontsize=14)
save_figure('cpi_score_boxplot.png')

# 8. Box Plot for Systolic and Diastolic Blood Pressure
plt.figure(figsize=(8, 5))
sns.boxplot(data=enhanced_dataset[['Systolic Blood Pressure', 'Diastolic Blood Pressure']], palette=['blue', 'red'])
plt.ylabel('Blood Pressure (mmHg)', fontsize=12)
plt.title('Box Plot of Systolic & Diastolic Blood Pressure', fontsize=14)
plt.xticks([0, 1], ['Systolic', 'Diastolic'])
save_figure('blood_pressure_boxplot.png')

# -----------------------------------------------------------------------------
# 6. CORRELATION ANALYSIS
# -----------------------------------------------------------------------------

print("\nPerforming correlation analysis...")

# Selecting relevant variables for correlation analysis
correlation_vars = ['Blood Sugar', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
                   'Hypertension', 'Cardiovascular Disease (Yes/No)', 'Periodontitis']

# Convert 'Hypertension' to binary (1 = Yes, 0 = No) for correlation analysis
enhanced_dataset['Hypertension'] = enhanced_dataset['Hypertension'].map({'Yes': 1, 'No': 0})

# Compute correlation matrix
correlation_matrix = enhanced_dataset[correlation_vars].corr()

# Plot heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
plt.title('Correlation Matrix of Systemic Health Indicators and Periodontitis', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
save_figure('correlation_matrix.png')

# Scatter Plots for Direct Comparisons
# 1. Blood Sugar vs. Periodontitis
plt.figure(figsize=(7, 5))
sns.scatterplot(data=enhanced_dataset, x='Blood Sugar', y='Periodontitis', alpha=0.6, color='purple')
plt.xlabel('Blood Sugar (mg/dL)', fontsize=12)
plt.ylabel('Periodontitis (0: No, 1: Yes)', fontsize=12)
plt.title('Scatter Plot: Blood Sugar vs. Periodontitis', fontsize=14)
plt.grid(True)
save_figure('blood_sugar_vs_periodontitis.png')

# 2. Systolic Blood Pressure vs. Periodontitis
plt.figure(figsize=(7, 5))
sns.scatterplot(data=enhanced_dataset, x='Systolic Blood Pressure', y='Periodontitis', alpha=0.6, color='blue')
plt.xlabel('Systolic Blood Pressure (mmHg)', fontsize=12)
plt.ylabel('Periodontitis (0: No, 1: Yes)', fontsize=12)
plt.title('Scatter Plot: Systolic Blood Pressure vs. Periodontitis', fontsize=14)
plt.grid(True)
save_figure('systolic_bp_vs_periodontitis.png')

# 3. Diastolic Blood Pressure vs. Periodontitis
plt.figure(figsize=(7, 5))
sns.scatterplot(data=enhanced_dataset, x='Diastolic Blood Pressure', y='Periodontitis', alpha=0.6, color='red')
plt.xlabel('Diastolic Blood Pressure (mmHg)', fontsize=12)
plt.ylabel('Periodontitis (0: No, 1: Yes)', fontsize=12)
plt.title('Scatter Plot: Diastolic Blood Pressure vs. Periodontitis', fontsize=14)
plt.grid(True)
save_figure('diastolic_bp_vs_periodontitis.png')

# Bar Charts for Categorical Data
# 4. Hypertension vs. Periodontitis
plt.figure(figsize=(7, 5))
sns.countplot(data=enhanced_dataset, x='Hypertension', hue='Periodontitis', palette=['gray', 'darkred'])
plt.xlabel('Hypertension (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Bar Chart: Hypertension vs. Periodontitis', fontsize=14)
plt.legend(title='Periodontitis', labels=['No', 'Yes'])
plt.grid(axis='y')
save_figure('hypertension_vs_periodontitis.png')

# 5. Cardiovascular Disease vs. Periodontitis
plt.figure(figsize=(7, 5))
sns.countplot(data=enhanced_dataset, x='Cardiovascular Disease (Yes/No)', hue='Periodontitis', palette=['gray', 'darkblue'])
plt.xlabel('Cardiovascular Disease (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Bar Chart: Cardiovascular Disease vs. Periodontitis', fontsize=14)
plt.legend(title='Periodontitis', labels=['No', 'Yes'])
plt.grid(axis='y')
save_figure('cvd_vs_periodontitis.png')

# -----------------------------------------------------------------------------
# 7. MACHINE LEARNING MODEL PREPARATION
# -----------------------------------------------------------------------------

print("\nPreparing data for machine learning models...")

# Selecting features and target variable
features = ['Blood Sugar', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
           'Hypertension', 'Cardiovascular Disease (Yes/No)', 'Oral Health Behavior Index',
           'Systemic Health Risk Score', 'Oral Disease Severity', 'Age']

target = 'Periodontitis'

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    enhanced_dataset[features],
    enhanced_dataset[target],
    test_size=0.2,
    random_state=42,
    stratify=enhanced_dataset[target]
)

# Store datasets for further use
ml_dataset = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Check for missing values in the training set
missing_values = ml_dataset["X_train"].isnull().sum()
missing_columns = missing_values[missing_values > 0]

if len(missing_columns) > 0:
    print("\nImputing missing values...")
    # Impute missing values in the 'Hypertension' column using the most frequent value (mode)
    ml_dataset["X_train"]["Hypertension"].fillna(ml_dataset["X_train"]["Hypertension"].mode()[0], inplace=True)
    ml_dataset["X_test"]["Hypertension"].fillna(ml_dataset["X_test"]["Hypertension"].mode()[0], inplace=True)
    print("Missing values imputation completed.")

# -----------------------------------------------------------------------------
# 8. MODEL TRAINING AND EVALUATION
# -----------------------------------------------------------------------------

print("\nTraining and evaluating machine learning models...")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Prepare dictionary for storing results
results = []

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    # Train model
    model.fit(ml_dataset["X_train"], ml_dataset["y_train"])

    # Predict on test set
    y_pred = model.predict(ml_dataset["X_test"])

    # Compute evaluation metrics
    accuracy = accuracy_score(ml_dataset["y_test"], y_pred)
    precision = precision_score(ml_dataset["y_test"], y_pred)
    recall = recall_score(ml_dataset["y_test"], y_pred)
    f1 = f1_score(ml_dataset["y_test"], y_pred)

    # Cross-validation score (5-fold)
    cv_scores = cross_val_score(model, ml_dataset["X_train"], ml_dataset["y_train"], cv=5)
    cv_mean = cv_scores.mean()

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "CV Score (Mean)": cv_mean
    })

# Convert results into a DataFrame for better visualization
results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df)

# Export results to CSV for reference
results_df.to_csv('model_performance_comparison.csv', index=False)

# -----------------------------------------------------------------------------
# 9. MODEL VISUALIZATION
# -----------------------------------------------------------------------------

print("\nGenerating model evaluation visualizations...")

# Colors for models
model_colors = sns.color_palette("Set2", len(models))

# ROC Curves for all models
plt.figure(figsize=(8, 6))

for (name, model), color in zip(models.items(), model_colors):
    # Get predicted probabilities
    y_prob = model.predict_proba(ml_dataset["X_test"])[:, 1]

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(ml_dataset["y_test"], y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], color="black", linestyle="--")

# Labels and title
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves with AUC Scores", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
save_figure('roc_curves.png')

# Bar plot comparing Accuracy, Precision, Recall, and F1 Score
plt.figure(figsize=(10, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Prepare data for bar plot
metrics_values = results_df.set_index("Model")[metrics].T

# Plot bar chart
metrics_values.plot(kind="bar", figsize=(10, 6), colormap="Set2", edgecolor="black")

# Labels and formatting
plt.xlabel("Evaluation Metrics", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Model Performance Comparison", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.legend(title="Model", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
save_figure('model_performance_comparison.png')

# Box plot of cross-validation scores
plt.figure(figsize=(8, 6))

# Prepare cross-validation scores in a structured format
cv_scores_list = []
model_names = []

for name, model in models.items():
    scores = cross_val_score(model, ml_dataset["X_train"], ml_dataset["y_train"], cv=5)
    cv_scores_list.extend(scores)
    model_names.extend([name] * len(scores))

# Create DataFrame for box plot
cv_df = pd.DataFrame({"Model": model_names, "Cross-Validation Score": cv_scores_list})

# Box plot
sns.boxplot(data=cv_df, x="Model", y="Cross-Validation Score", palette="Set2")

# Labels and title
plt.xticks(rotation=45, fontsize=10)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Cross-Validation Score", fontsize=12)
plt.title("Cross-Validation Score Distribution", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
save_figure('cv_score_distribution.png')

# -----------------------------------------------------------------------------
# 10. HYPERPARAMETER TUNING FOR BEST MODELS
# -----------------------------------------------------------------------------

print("\nPerforming hyperparameter tuning for best models...")

# Define hyperparameter grids for tuning
# Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with 5-Fold Cross Validation
print("Tuning Random Forest...")
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                             rf_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
rf_grid_search.fit(ml_dataset["X_train"], ml_dataset["y_train"])

print("Tuning Gradient Boosting...")
gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42),
                             gb_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
gb_grid_search.fit(ml_dataset["X_train"], ml_dataset["y_train"])

# Best parameters and best scores
rf_best_params = rf_grid_search.best_params_
rf_best_score = rf_grid_search.best_score_

gb_best_params = gb_grid_search.best_params_
gb_best_score = gb_grid_search.best_score_

print("\nRandom Forest best parameters:", rf_best_params)
print("Random Forest best F1 score:", rf_best_score)

print("\nGradient Boosting best parameters:", gb_best_params)
print("Gradient Boosting best F1 score:", gb_best_score)

# -----------------------------------------------------------------------------
# 11. FEATURE IMPORTANCE
# -----------------------------------------------------------------------------

# Select the best model based on CV score
best_model = None
best_model_name = None
best_score = 0

for result in results:
    if result["CV Score (Mean)"] > best_score:
        best_score = result["CV Score (Mean)"]
        best_model_name = result["Model"]

print(f"\nBest model based on cross-validation: {best_model_name} (CV Score: {best_score:.4f})")

# Get the best model instance
if best_model_name == "Random Forest":
    # Use tuned Random Forest
    best_model = RandomForestClassifier(**rf_best_params, random_state=42)
elif best_model_name == "Gradient Boosting":
    # Use tuned Gradient Boosting
    best_model = GradientBoostingClassifier(**gb_best_params, random_state=42)
else:
    # Use the original best model
    best_model = models[best_model_name]

# Train the best model
best_model.fit(ml_dataset["X_train"], ml_dataset["y_train"])

# Feature importance analysis (if applicable)
if hasattr(best_model, 'feature_importances_'):
    # Get feature importances
    importances = best_model.feature_importances_
    # Create a DataFrame with features and their importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title(f'Feature Importance from {best_model_name}', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    save_figure('feature_importance.png')
    
    print("\nFeature Importance Ranking:")
    print(feature_importance)
    
    # Export feature importance to CSV
    feature_importance.to_csv('feature_importance.csv', index=False)

print("\nAnalysis completed. All visualizations and results have been saved.")
