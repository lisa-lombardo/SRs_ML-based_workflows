import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def sanitize_and_convert(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"Sanitization error for {smiles}: {e}")
        return None
    return mol

def compute_descriptors(df):
    df['mols'] = df['canonical_smiles'].apply(sanitize_and_convert)
    df = df[df['mols'].notnull()].reset_index(drop=True)
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_df = calc.pandas(df['mols'])
    return pd.concat([df, mordred_df], axis=1)

def map_selectivity_label(row):
    if row['selectivity'] == 'S1R selective':
        return 1
    elif row['selectivity'] == 'S2R selective':
        return 2
    elif row['selectivity'] in ['semi-selective', 'dual binder']:
        return 0
    else:
        return 4

def evaluate_predictions(df, pred_s1r, pred_s2r, threshold=5.0):
    pred_s1r_bin = (pred_s1r > threshold).astype(int)
    pred_s2r_bin = (pred_s2r > threshold).astype(int)

    # Compute selectivity class
    selectivity_class = np.full(len(pred_s1r), 0)
    for i in range(len(pred_s1r)):
        if pred_s1r[i] > threshold and pred_s2r[i] <= threshold and (pred_s1r[i] - pred_s2r[i]) >= 2:
            selectivity_class[i] = 1
        elif pred_s2r[i] > threshold and pred_s1r[i] <= threshold and (pred_s2r[i] - pred_s1r[i]) >= 2:
            selectivity_class[i] = 2
        elif pred_s1r[i] <= threshold and pred_s2r[i] <= threshold:
            selectivity_class[i] = 4

    y_true = df['selectivity_label']
    print("\nClassification Report for Selectivity:")
    print(classification_report(y_true, selectivity_class, target_names=['non-selective', 'S1R selective', 'S2R selective', 'non-binder']))

    conf_matrix = confusion_matrix(y_true, selectivity_class, labels=[0, 1, 2, 4])
    print(f"Confusion Matrix:\n{conf_matrix}")

    accuracy = accuracy_score(y_true, selectivity_class)
    f1 = f1_score(y_true, selectivity_class, average='weighted')
    precision = precision_score(y_true, selectivity_class, average='weighted')
    recall = recall_score(y_true, selectivity_class, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")

def main(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    df['selectivity_label'] = df.apply(map_selectivity_label, axis=1)

    df_desc = compute_descriptors(df)
    feature_cols = df_desc.columns.difference(['pActivity', 'activity', 'selectivity', 'activity S1R', 'activity S2R', 'selectivity_label', 'mols', 'canonical_smiles'])
    df_features = df_desc[feature_cols]

    # Load selected features
    selected_features_S1R = joblib.load('../../models/regression/selected_features_s1r.joblib')
    selected_features_S2R = joblib.load('../../models/regression/selected_features_s2r.joblib')
    X_S1R = df_features[selected_features_S1R]
    X_S2R = df_features[selected_features_S2R]

    # Load pipelines
    model_S1R = joblib.load('../../models/regression/xgb_final_pipeline_s1r.joblib')
    model_S2R = joblib.load('../../models/regression/et_final_pipeline_s2r.joblib')

    # Predict
    preds_S1R = model_S1R.predict(X_S1R)
    preds_S2R = model_S2R.predict(X_S2R)

    # Evaluate
    evaluate_predictions(df_desc, preds_S1R, preds_S2R)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and evaluate S1R/S2R selectivity based on regression models.")
    parser.add_argument("--input", required=True, help="Path to input CSV with SMILES and labels")
    parser.add_argument("--output", required=True, help="Directory to store output")
    args = parser.parse_args()

    main(args.input, args.output)