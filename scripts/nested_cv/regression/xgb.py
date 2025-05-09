#!/usr/bin/env python

import numpy as np
import pandas as pd
import cupy as cp
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Nested CV for XGBRegressor on multiple descriptor sets.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input descriptor CSVs')
    parser.add_argument('--output_dir', type=str, default='out', help='Directory to save the results CSV')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    # Define RMSE as a scorer for cross-validation
    def rmse_scorer(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Load the datasets
    datasets = {
        'rdkit': pd.read_csv('../df_rdkit.csv'),
        'mordred': pd.read_csv('../df_mordred.csv'),
        'morgan_2': pd.read_csv('../df_morgan_2_2048.csv'),
        'morgan_3': pd.read_csv('../df_morgan_3_2048.csv'),
        'maccs': pd.read_csv('../df_maccs.csv')
    }

    # DataFrame to store inner and outer results for each descriptor
    results_df = pd.DataFrame(columns=['Descriptor', 'Outer Fold', 'Best RMSE (inner test set)', 'Best R² (inner test set)', 'Best Parameters', 'RMSE (outer test set)', 'R² (outer test set)', 'Mean RMSE (outer loop)', 'Mean R² (outer loop)'])

    # Loop through each dataset to perform nested cross-validation
    for descriptor_name, df in datasets.items():
        print(f"\nXGB: Performing nested cross-validation for {descriptor_name} descriptor set...")
        
        # Define features and target
        X = df.drop(columns=['pActivity', 'selectivity', 'activity']).values
        y = df['pActivity'].values
        
        # Standardize features on CPU, then transfer to GPU
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert X_scaled and y to GPU using CuPy
        X_scaled = cp.array(X_scaled)
        y = cp.array(y)

        # Outer Cross-Validation (5 folds)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
        
        # Outer loop: Iterate through each of the 5 outer folds
        outer_rmse_scores = []
        outer_r2_scores = []
        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_scaled), start=1):
            print(f"\nOuter fold {outer_fold}")
            
            # Split the data for outer folds (remains on GPU)
            X_train_outer, X_test_outer = X_scaled[outer_train_idx], X_scaled[outer_test_idx]
            y_train_outer, y_test_outer = y[outer_train_idx], y[outer_test_idx]
            
            # Convert GPU data to CPU for cross-validation purposes (inner cross-validation)
            X_train_outer_cpu = X_train_outer.get()
            y_train_outer_cpu = y_train_outer.get()
            
            # Inner Cross-Validation (for hyperparameter optimization)
            def objective(trial):
                # Suggest hyperparameters for XGB
                n_estimators = trial.suggest_int('n_estimators', 100, 900)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

                # Create the XGBRegressor with suggested hyperparameters (using GPU)
                xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, 
                                   subsample=subsample, colsample_bytree=colsample_bytree, 
                                   device='cuda')
                
                # Perform cross-validation on CPU arrays (cross_val_score uses CPU-based arrays)
                rmse_results = []
                r2_results = []
                inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
                for train_idx, val_idx in inner_cv.split(X_train_outer_cpu):
                    X_train_inner, X_val_inner = X_train_outer_cpu[train_idx], X_train_outer_cpu[val_idx]
                    y_train_inner, y_val_inner = y_train_outer_cpu[train_idx], y_train_outer_cpu[val_idx]
                    
                    # Train on GPU
                    xgb.fit(cp.array(X_train_inner), cp.array(y_train_inner))
                    # Predict on GPU
                    y_val_pred = xgb.predict(cp.array(X_val_inner))
                    
                    # Calculate metrics (on CPU)
                    rmse = rmse_scorer(y_val_inner, cp.asnumpy(y_val_pred))
                    r2 = r2_score(y_val_inner, cp.asnumpy(y_val_pred))
                    
                    rmse_results.append(rmse)
                    r2_results.append(r2)
                
                avg_rmse = np.mean(rmse_results)
                avg_r2 = np.mean(r2_results)
                
                trial.set_user_attr('r2', avg_r2)  # Store R² for this trial
                
                return avg_rmse
            
            # Optuna hyperparameter optimization within the outer training set
            study = optuna.create_study(direction='minimize', sampler=TPESampler())
            study.optimize(objective, n_trials=50)
            
            # Best hyperparameters and R² found during inner cross-validation
            best_params = study.best_params
            best_rmse_inner = study.best_value  # Best RMSE from inner cross-validation
            best_r2_inner = study.best_trial.user_attrs['r2']  # Best R² from inner cross-validation
            print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")
            print(f"Best RMSE (avg of inner folds): {best_rmse_inner:.4f}")
            print(f"Best R² (avg of inner folds): {best_r2_inner:.4f}")
            
            # Train the final model on the outer training set using the best hyperparameters (using GPU)
            best_model = XGBRegressor(**best_params, device='cuda')
            best_model.fit(X_train_outer, y_train_outer)  # Training using GPU data
            
            # Predict using GPU data (no data movement back to CPU)
            y_test_pred = best_model.predict(X_test_outer)  # This will be a numpy.ndarray

            # Move GPU target data to CPU for scoring
            y_test_outer_cpu = y_test_outer.get()

            # Calculate the RMSE and R² scores on CPU
            rmse_outer = rmse_scorer(y_test_outer_cpu, y_test_pred)  # Using NumPy arrays for scoring
            r2_outer = r2_score(y_test_outer_cpu, y_test_pred)  # Using NumPy arrays for scoring

            print(f"RMSE on outer test set: {rmse_outer:.4f}")
            print(f"R² on outer test set: {r2_outer:.4f}")
            
            # Append outer fold RMSE and R² scores to lists
            outer_rmse_scores.append(rmse_outer)
            outer_r2_scores.append(r2_outer)
            
            # Store results for this outer fold
            fold_result = pd.DataFrame({
                'Descriptor': [descriptor_name],
                'Outer Fold': [outer_fold],
                'Best RMSE (inner test set)': [best_rmse_inner],
                'Best R² (inner test set)': [best_r2_inner],
                'Best Parameters': [str(best_params)],
                'RMSE (outer test set)': [rmse_outer],
                'R² (outer test set)': [r2_outer],
                'Mean RMSE (outer loop)': [None],  # Will be updated after all folds are done
                'Mean R² (outer loop)': [None]     # Will be updated after all folds are done
            })
            results_df = pd.concat([results_df, fold_result], ignore_index=True)
        
        # Calculate the overall performance (mean and std of outer test set RMSE and R²)
        mean_rmse_outer = np.mean(outer_rmse_scores)
        std_rmse_outer = np.std(outer_rmse_scores)
        mean_r2_outer = np.mean(outer_r2_scores)
        std_r2_outer = np.std(outer_r2_scores)
        
        print(f"\nOverall Performance for {descriptor_name}:")
        print(f"Mean RMSE (outer): {mean_rmse_outer:.4f} ± {std_rmse_outer:.4f}")
        print(f"Mean R² (outer): {mean_r2_outer:.4f} ± {std_r2_outer:.4f}\n")
        
        # Update the "Mean RMSE (outer loop)" and "Mean R² (outer loop)" columns for all outer folds of this descriptor
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean RMSE (outer loop)'] = mean_rmse_outer
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean R² (outer loop)'] = mean_r2_outer
        results_df = results_df.round(3)

    # Save the results to a CSV file
    results_filename = "out/XGB_nested_cv_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")


if __name__ == '__main__':
    main()
