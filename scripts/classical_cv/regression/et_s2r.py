#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
import optuna
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import joblib
import os

# Ensure output directories exist
output_dir = "../../../models/regression"
os.makedirs(output_dir, exist_ok=True)

# Define RMSE as a scorer for cross-validation
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load dataset
df = pd.read_csv('../../../data/features/dataset_2/df_mordred_regression.csv')

# Define features and target
X = df.drop(columns=['pActivity', 'selectivity', 'activity'])
y = df['pActivity']

# Save the feature names for future use
feature_names = X.columns
feature_names_filename = f"{output_dir}/selected_features.joblib"
joblib.dump(feature_names, feature_names_filename)
print(f"Feature names saved as {feature_names_filename}")

# DataFrame to store results for cross-validation
results_df = pd.DataFrame(columns=['Mean RMSE', 'Std RMSE', 'Mean R²', 'Std R²', 'Best Parameters'])

# Split the data using 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=1)

# Hyperparameter optimization using Optuna
def objective(trial):
    # Suggest hyperparameters for ExtraTreesRegressor
    bootstrap = False  # not "False"
    max_features = 'sqrt'  # or trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    n_estimators = trial.suggest_int('n_estimators', 445, 995)
    max_depth = trial.suggest_int('max_depth', 23, 48)
    min_samples_split = trial.suggest_int('min_samples_split', 9, 13)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)  # optional tuning
    criterion = trial.suggest_categorical('criterion', ["squared_error", "friedman_mse", "poisson"])


    # Create the ExtraTreesRegressor with suggested hyperparameters
    regressor = ExtraTreesRegressor(bootstrap=bootstrap, max_features=max_features, n_estimators=n_estimators, 
                                      max_depth=max_depth, min_samples_split=min_samples_split, 
                                      min_samples_leaf=min_samples_leaf, criterion=criterion, n_jobs=-1)

    pipeline =  Pipeline([('scaler', StandardScaler()), ('regressor', regressor)])	

    # Perform cross-validation and return the average RMSE
    rmse_results = cross_val_score(pipeline, X, y, cv=cv, scoring=make_scorer(rmse_scorer))
    avg_rmse = np.mean(rmse_results)
    
    return avg_rmse

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

# Best hyperparameters found during cross-validation
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Create the final model with the best hyperparameters
best_model = ExtraTreesRegressor(**best_params)
pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', best_model)])

# Evaluate the final model using 5-fold cross-validation
rmse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=make_scorer(rmse_scorer))
r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")

# Store the results
results_df = pd.concat([results_df, pd.DataFrame({
    'Mean RMSE': [mean_rmse],
    'Std RMSE': [std_rmse],
    'Mean R²': [mean_r2],
    'Std R²': [std_r2],
    'Best Parameters': [str(best_params)]
})], ignore_index=True)

# Save the final pipeline (scaler + model)
pipeline.fit(X, y)
joblib.dump(pipeline, os.path.join(output_dir, "et_final_pipeline_s2r.joblib"))
print(f"Final model saved to {output_dir}/et_final_pipeline_s2r.joblib")

# Save the cross-validation results to a CSV file
results_filename = f"{output_dir}/cv_results_summary.csv"
results_df.to_csv(results_filename, index=False)
print(f"\nResults saved to {results_filename}")
