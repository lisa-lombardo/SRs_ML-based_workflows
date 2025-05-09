#!/usr/bin/env python

import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, matthews_corrcoef, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import optuna
from optuna.samplers import TPESampler
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Nested CV for KNeighborsClassifier on multiple descriptor sets.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input descriptor CSVs')
    parser.add_argument('--output_dir', type=str, default='out', help='Directory to save the results CSV')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load the datasets
    datasets = {
        'rdkit': pd.read_csv('../df_rdkit.csv'),
        'mordred': pd.read_csv('../df_mordred.csv'),
        'morgan_2': pd.read_csv('../df_ECFP4.csv'),
        'morgan_3': pd.read_csv('../df_ECFP6.csv'),
        'maccs': pd.read_csv('../df_maccs.csv')
    }

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'Descriptor', 'Outer Fold', 'Best AUC_w (inner test set)', 
        'Best F1_w (inner test set)', 'Best Parameters', 
        'AUC_w (outer test set)', 'F1_w (outer test set)', 
        'Precision (outer test set)', 'Recall (outer test set)', 
        'MCC (outer test set)', 'Mean AUC_w (outer loop)', 
        'Mean F1_w (outer loop)', 'Mean Precision (outer loop)', 
        'Mean Recall (outer loop)', 'Mean MCC (outer loop)'
    ])

    # Loop through each dataset to perform nested cross-validation
    for descriptor_name, df in datasets.items():
        print(f"\nKNN: Performing nested cross-validation for {descriptor_name} descriptor set...")

         # Define features and target
        X = df.drop(columns=[ 'pActivity', 'activity', 'selectivity', 'class_label'])
        y = df['class_label']

        # Outer cross-validation setup
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        outer_auc_scores, outer_f1_scores, outer_precision_scores = [], [], []
        outer_recall_scores, outer_mcc_scores = [], []

        outer_fold = 0

        # Outer loop for evaluating the model on the test set
        for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
            outer_fold += 1
            print(f"\nOuter fold {outer_fold} for {descriptor_name}")

            # Split data for the current outer fold
            X_train_outer, X_test_outer = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
            y_train_outer, y_test_outer = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

            # Apply Random Under Sampling to balance the training data
            rus = RandomUnderSampler(random_state=1)
            X_train_outer_resampled, y_train_outer_resampled = rus.fit_resample(X_train_outer, y_train_outer)

            # Standardize features within the fold
            scaler = StandardScaler()
            X_train_outer_scaled = scaler.fit_transform(X_train_outer_resampled)
            X_test_outer_scaled = scaler.transform(X_test_outer)

            # Inner Cross-Validation (for hyperparameter optimization)
            def objective(trial, X_resampled, y_resampled):
                # Suggest hyperparameters for KNeighborsClassifier
                n_neighbors = trial.suggest_int('n_neighbors', 2, 30)
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                metric = trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan'])
                algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
                leaf_size = trial.suggest_int('leaf_size', 1, 50)

                # Create the KNeighborsClassifier with suggested hyperparameters
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm=algorithm, leaf_size=leaf_size)

                # Inner cross-validation with undersampling applied to each inner fold
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                auc_scores = []

                # We need to use integer-based indices for the resampled dataset
                X_resampled = X_resampled.reset_index(drop=True)
                y_resampled = y_resampled.reset_index(drop=True)

                for inner_train_idx, inner_test_idx in inner_cv.split(X_resampled, y_resampled):
                    # Use iloc for integer-based indexing with inner_train_idx and inner_test_idx
                    X_train_inner, X_valid_inner = X_resampled.iloc[inner_train_idx], X_resampled.iloc[inner_test_idx]
                    y_train_inner, y_valid_inner = y_resampled.iloc[inner_train_idx], y_resampled.iloc[inner_test_idx]

                    # Standardize features
                    X_train_inner_scaled = scaler.fit_transform(X_train_inner)
                    X_valid_inner_scaled = scaler.transform(X_valid_inner)

                    knn.fit(X_train_inner_scaled, y_train_inner)
                    y_pred_inner = knn.predict_proba(X_valid_inner_scaled)

                    # Calculate weighted AUC for the validation set
                    auc_score = roc_auc_score(y_valid_inner, y_pred_inner, average='weighted', multi_class='ovr')
                    auc_scores.append(auc_score)

                avg_auc_w = np.mean(auc_scores)
                trial.set_user_attr('auc_weighted', avg_auc_w)
                return avg_auc_w  # Optimize weighted AUC

            # Optuna study for hyperparameter tuning
            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(lambda trial: objective(trial, X_train_outer_resampled, y_train_outer_resampled), n_trials=50)

            # Best hyperparameters and metrics from inner cross-validation
            best_params = study.best_params
            best_auc_w_inner = study.best_value
            print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")
            print(f"Best AUC_W (inner test set): {best_auc_w_inner:.4f}")

            # Train the final model on the outer training set using the best hyperparameters
            best_model = KNeighborsClassifier(**best_params)
            best_model.fit(X_train_outer_scaled, y_train_outer_resampled)

            # Evaluate the model on the outer test set
            y_pred_outer = best_model.predict(X_test_outer_scaled)
            y_proba_outer = best_model.predict_proba(X_test_outer_scaled)

            # Calculate metrics for the outer test set
            auc_weighted_outer = roc_auc_score(y_test_outer, y_proba_outer, average='weighted', multi_class='ovr')
            f1_weighted_outer = f1_score(y_test_outer, y_pred_outer, average='weighted')
            precision_outer = precision_score(y_test_outer, y_pred_outer, average='weighted')
            recall_outer = recall_score(y_test_outer, y_pred_outer, average='weighted')
            mcc_outer = matthews_corrcoef(y_test_outer, y_pred_outer)

            # Print results for the outer test set
            print(f"AUC_W (outer test set): {auc_weighted_outer:.4f}")
            print(f"F1_W (outer test set): {f1_weighted_outer:.4f}")
            print(f"Precision (outer test set): {precision_outer:.4f}")
            print(f"Recall (outer test set): {recall_outer:.4f}")
            print(f"MCC (outer test set): {mcc_outer:.4f}")

            # Store metrics
            outer_auc_scores.append(auc_weighted_outer)
            outer_f1_scores.append(f1_weighted_outer)
            outer_precision_scores.append(precision_outer)
            outer_recall_scores.append(recall_outer)
            outer_mcc_scores.append(mcc_outer)

            # Append results to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame({
                'Descriptor': [descriptor_name], 
                'Outer Fold': [outer_fold],
                'Best AUC_w (inner test set)': [best_auc_w_inner],
                'Best F1_w (inner test set)': [f1_weighted_outer],
                'Best Parameters': [str(best_params)],
                'AUC_w (outer test set)': [auc_weighted_outer],
                'F1_w (outer test set)': [f1_weighted_outer],
                'Precision (outer test set)': [precision_outer],
                'Recall (outer test set)': [recall_outer],
                'MCC (outer test set)': [mcc_outer]
            })], ignore_index=True)

        # Compute mean and std for each outer metric
        mean_auc_outer = np.mean(outer_auc_scores)
        std_auc_outer = np.std(outer_auc_scores)
        mean_f1_outer = np.mean(outer_f1_scores)
        std_f1_outer = np.std(outer_f1_scores)
        mean_precision_outer = np.mean(outer_precision_scores)
        std_precision_outer = np.std(outer_precision_scores)
        mean_recall_outer = np.mean(outer_recall_scores)
        std_recall_outer = np.std(outer_recall_scores)
        mean_mcc_outer = np.mean(outer_mcc_scores)
        std_mcc_outer = np.std(outer_mcc_scores)

        print(f"\nOverall Performance for {descriptor_name}:")
        print(f"Mean AUC_W (outer): {mean_auc_outer:.4f} ± {std_auc_outer:.4f}")
        print(f"Mean F1_W (outer): {mean_f1_outer:.4f} ± {std_f1_outer:.4f}")
        print(f"Mean Precision (outer): {mean_precision_outer:.4f} ± {std_precision_outer:.4f}")
        print(f"Mean Recall (outer): {mean_recall_outer:.4f} ± {std_recall_outer:.4f}")
        print(f"Mean MCC (outer): {mean_mcc_outer:.4f} ± {std_mcc_outer:.4f}\n")

        # Update the "Mean" columns in the DataFrame
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean AUC_w (outer loop)'] = mean_auc_outer
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean F1_w (outer loop)'] = mean_f1_outer
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean Precision (outer loop)'] = mean_precision_outer
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean Recall (outer loop)'] = mean_recall_outer
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean MCC (outer loop)'] = mean_mcc_outer

    # Save the results to a CSV file
    results_filename = "out/nested_cv_results_summary.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")



if __name__ == '__main__':
    main()