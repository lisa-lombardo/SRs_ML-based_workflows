#!/usr/bin/env python 

import pandas as pd 
import numpy as np
import os
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import optuna
from optuna.samplers import TPESampler

def main():
    parser = argparse.ArgumentParser(description="Nested CV for KNeighborsClassifier on multiple descriptor sets.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input descriptor CSVs')
    parser.add_argument('--output_dir', type=str, default='out', help='Directory to save the results CSV')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        'rdkit': pd.read_csv(os.path.join(input_dir, 'df_rdkit.csv')),
        'mordred': pd.read_csv(os.path.join(input_dir, 'df_mordred.csv')),
        'morgan_2': pd.read_csv(os.path.join(input_dir, 'df_ECFP4.csv')),
        'morgan_3': pd.read_csv(os.path.join(input_dir, 'df_ECFP6.csv')),
        'maccs': pd.read_csv(os.path.join(input_dir, 'df_maccs.csv'))
    }

    results_df = pd.DataFrame(columns=[
        'Descriptor', 'Outer Fold', 'Best AUC (inner test set)', 
        'Best F1 (inner test set)', 'Best Parameters', 
        'AUC (outer test set)', 'F1 (outer test set)', 
        'Precision (outer test set)', 'Recall (outer test set)', 
        'MCC (outer test set)', 'Mean AUC (outer loop)', 
        'Mean F1 (outer loop)', 'Mean Precision (outer loop)', 
        'Mean Recall (outer loop)', 'Mean MCC (outer loop)'
    ])

    for descriptor_name, df in datasets.items():
        print(f"\nKNN: Performing nested cross-validation for {descriptor_name} descriptor set...")

        X = df.drop(columns=['selectivity', 'pActivity', 'activity', 'labels'])
        y = df['labels']

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        outer_auc_scores, outer_f1_scores = [], []
        outer_precision_scores, outer_recall_scores, outer_mcc_scores = [], [], []

        outer_fold = 0

        for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
            outer_fold += 1
            print(f"\nOuter fold {outer_fold} for {descriptor_name}")

            X_train_outer, X_test_outer = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
            y_train_outer, y_test_outer = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

            rus = RandomUnderSampler(random_state=1)
            X_train_outer_resampled, y_train_outer_resampled = rus.fit_resample(X_train_outer, y_train_outer)

            scaler = StandardScaler()
            X_train_outer_scaled = scaler.fit_transform(X_train_outer_resampled)
            X_test_outer_scaled = scaler.transform(X_test_outer)

            def objective(trial, X_resampled, y_resampled):
                n_neighbors = trial.suggest_int('n_neighbors', 2, 30)
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                metric = trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan'])
                algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
                leaf_size = trial.suggest_int('leaf_size', 1, 50)

                knn = KNeighborsClassifier(
                    n_neighbors=n_neighbors, weights=weights, metric=metric,
                    algorithm=algorithm, leaf_size=leaf_size
                )

                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                auc_scores = []

                X_resampled = X_resampled.reset_index(drop=True)
                y_resampled = y_resampled.reset_index(drop=True)

                for inner_train_idx, inner_test_idx in inner_cv.split(X_resampled, y_resampled):
                    X_train_inner = X_resampled.iloc[inner_train_idx]
                    X_valid_inner = X_resampled.iloc[inner_test_idx]
                    y_train_inner = y_resampled.iloc[inner_train_idx]
                    y_valid_inner = y_resampled.iloc[inner_test_idx]

                    X_train_inner_scaled = scaler.fit_transform(X_train_inner)
                    X_valid_inner_scaled = scaler.transform(X_valid_inner)

                    knn.fit(X_train_inner_scaled, y_train_inner)
                    y_pred_proba_inner = knn.predict_proba(X_valid_inner_scaled)[:, 1]
                    auc_scores.append(roc_auc_score(y_valid_inner, y_pred_proba_inner))

                avg_auc = np.mean(auc_scores)
                trial.set_user_attr('auc', avg_auc)
                return avg_auc

            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(lambda trial: objective(trial, X_train_outer_resampled, y_train_outer_resampled), n_trials=50)

            best_params = study.best_params
            best_auc_inner = study.best_value
            print(f"Best hyperparameters for outer fold {outer_fold}: {best_params}")
            print(f"Best AUC (inner test set): {best_auc_inner:.4f}")

            best_model = KNeighborsClassifier(**best_params)
            best_model.fit(X_train_outer_scaled, y_train_outer_resampled)

            y_pred_outer = best_model.predict(X_test_outer_scaled)
            y_proba_outer = best_model.predict_proba(X_test_outer_scaled)[:, 1]

            auc_outer = roc_auc_score(y_test_outer, y_proba_outer)
            f1_outer = f1_score(y_test_outer, y_pred_outer)
            precision_outer = precision_score(y_test_outer, y_pred_outer)
            recall_outer = recall_score(y_test_outer, y_pred_outer)
            mcc_outer = matthews_corrcoef(y_test_outer, y_pred_outer)

            print(f"AUC (outer test set): {auc_outer:.4f}")
            print(f"F1 (outer test set): {f1_outer:.4f}")
            print(f"Precision (outer test set): {precision_outer:.4f}")
            print(f"Recall (outer test set): {recall_outer:.4f}")
            print(f"MCC (outer test set): {mcc_outer:.4f}")

            outer_auc_scores.append(auc_outer)
            outer_f1_scores.append(f1_outer)
            outer_precision_scores.append(precision_outer)
            outer_recall_scores.append(recall_outer)
            outer_mcc_scores.append(mcc_outer)

            results_df = pd.concat([results_df, pd.DataFrame({
                'Descriptor': [descriptor_name], 
                'Outer Fold': [outer_fold],
                'Best AUC (inner test set)': [best_auc_inner],
                'Best F1 (inner test set)': [f1_outer],
                'Best Parameters': [str(best_params)],
                'AUC (outer test set)': [auc_outer],
                'F1 (outer test set)': [f1_outer],
                'Precision (outer test set)': [precision_outer],
                'Recall (outer test set)': [recall_outer],
                'MCC (outer test set)': [mcc_outer]
            })], ignore_index=True)

        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean AUC (outer loop)'] = np.mean(outer_auc_scores)
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean F1 (outer loop)'] = np.mean(outer_f1_scores)
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean Precision (outer loop)'] = np.mean(outer_precision_scores)
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean Recall (outer loop)'] = np.mean(outer_recall_scores)
        results_df.loc[results_df['Descriptor'] == descriptor_name, 'Mean MCC (outer loop)'] = np.mean(outer_mcc_scores)

    results_filename = os.path.join(output_dir, "nested_cv_results_summary.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")

if __name__ == '__main__':
    main()
