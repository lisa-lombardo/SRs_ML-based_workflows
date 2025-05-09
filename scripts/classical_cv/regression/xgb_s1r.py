import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import joblib
import argparse

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    parser = argparse.ArgumentParser(description="XGBoost Regression with Optuna and CV")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='out_XGB', help='Directory to save outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.input_file)

    # Define features and target
    X = df.drop(columns=['pActivity', 'selectivity', 'activity'])
    y = df['pActivity']

    # Save the feature names for future use
    feature_names = X.columns
    feature_names_filename = f"{args.output_dir}/selected_features.joblib"
    joblib.dump(feature_names, feature_names_filename)
    print(f"Feature names saved as {feature_names_filename}")

    # Define cross-validation setup
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 615, 819)
        max_depth = trial.suggest_int('max_depth', 6, 9)
        learning_rate = trial.suggest_float('learning_rate', 0.017, 0.04)
        subsample = trial.suggest_float('subsample', 0.54, 0.96)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.58, 0.99)

        regressor = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                 learning_rate=learning_rate, subsample=subsample, 
                                 colsample_bytree=colsample_bytree, n_jobs=-1)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])

        rmse_results = cross_val_score(pipeline, X, y, cv=kfold, scoring=make_scorer(rmse_scorer))
        return np.mean(rmse_results)

    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print(f"Best hyperparameters from Optuna: {best_params}")

    results_df = pd.DataFrame(columns=['Fold', 'RMSE', 'R2'])
    best_model = XGBRegressor(**best_params, n_jobs=-1)

    fold = 0
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in kfold.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', best_model)
        ])

        pipeline.fit(X_train, y_train)

        model_filename = f"{args.output_dir}/xgb_model_fold_{fold}.joblib"
        joblib.dump(pipeline, model_filename)
        print(f"Pipeline for fold {fold} saved as {model_filename}")

        y_pred = pipeline.predict(X_test)
        rmse = rmse_scorer(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

        results_df.loc[len(results_df)] = [fold, rmse, r2]

    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print(f"\nOverall Performance:")
    print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}\n")

    results_filename = f"{args.output_dir}/XGBR_classical_cv_results_summary.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', best_model)
    ])
    final_pipeline.fit(X, y)

    final_model_filename = f"{args.output_dir}/XGBR_final_pipeline.joblib"
    joblib.dump(final_pipeline, final_model_filename)
    print(f"Final pipeline saved as {final_model_filename}")

if __name__ == "__main__":
    main()
