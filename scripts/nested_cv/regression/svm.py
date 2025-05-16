#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold,train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import joblib
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Nested CV for SVR on multiple descriptor sets.")
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
    results_df = pd.DataFrame(columns=['Descriptor', 'Outer Fold', 'Best RMSE (inner test set)', 'Best R² (inner test set)', 
                                       'Best Parameters', 'RMSE (outer test set)', 'R² (outer test set)', 
                                       'Mean RMSE (outer loop)', 'Mean R² (outer loop)'])

    # Loop through each dataset to perform nested cross-validation
    for descriptor_name, df in datasets.items():
        print(f"\nPerforming nested cross-validation for {descriptor_name} descriptor set...")
        
        # Define features and target
        X = df.drop(columns=['pActivity', 'selectivity', 'activity'])
        y = df['pActivity']
        
        # Outer Cross-Validation (5 folds)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
        
        # Outer loop: Iterate through each of the 5 outer folds
        outer_rmse_scores = []
        outer_r2_scores = []
        
        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y), start=1):
            print(f"\nOuter fold {outer_fold}")
            
            X_train_outer, X_test_outer = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
            y_train_outer, y_test_outer = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

            # Inner Cross-Validation (for hyperparameter optimization)
            def objective(trial):
                # Suggest hyperparameters for SVR
                C = trial.suggest_float('C', 0.01, 100.0, log=True)
                gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
                epsilon = trial.suggest_float('epsilon', 1e-3, 1.0, log=True)

                # Create the SVR with suggested hyperparameters
                svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            
                pipeline = Pipeline([('scaler', StandardScaler()), ('svr', svr)])
            
                inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)

                # Perform inner cross-validation and calculate both RMSE and R²
                rmse_results = cross_val_score(pipeline, X_train_outer, y_train_outer, cv=inner_cv, scoring=make_scorer(rmse_scorer))
                r2_results = cross_val_score(pipeline, X_train_outer, y_train_outer, cv=inner_cv, scoring='r2')
                
                avg_rmse = np.mean(rmse_results)
                avg_r2 = np.mean(r2_results)
                
                trial.set_user_attr('r2', avg_r2)  # Store R² for this trial
                
                return avg_rmse
            
            # Optuna hyperparameter optimization within the outer training set
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=50)
            
            # Best hyperparameters and R² found during inner cross-validation
            best_params = study.best_params
            best_rmse_inner = study.best_value  # Best RMSE from inner cross-validation
            best_r2_inner = study.best_trial.user_attrs['r2']  # Best R² from inner cross-validation
            print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")
            print(f"Best RMSE (avg of inner folds): {best_rmse_inner:.4f}")
            print(f"Best R² (avg of inner folds): {best_r2_inner:.4f}")
            
            # Train the final model on the outer training set using the best hyperparameters
            best_model = SVR(
                    C=best_params['C'], 
                    gamma=best_params['gamma'], 
                    kernel='rbf', 
                    epsilon=best_params['epsilon']
            )

            pipeline = Pipeline([('scaler', StandardScaler()), ('svr', best_model)])
            pipeline.fit(X_train_outer, y_train_outer)
            
            # Evaluate on the outer test set
            y_test_pred = pipeline.predict(X_test_outer)
            rmse_outer = rmse_scorer(y_test_outer, y_test_pred)
            r2_outer = r2_score(y_test_outer, y_test_pred)
            
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

    # Save the results to a CSV file
    results_filename = "out/nested_cv_results_summary_with_r2.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")


if __name__ == '__main__':
    main()
