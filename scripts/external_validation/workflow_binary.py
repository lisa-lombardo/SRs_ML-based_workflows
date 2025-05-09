import os
import argparse
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef


def sanitize_and_compute_descriptors(df):
    mols = [Chem.MolFromSmiles(smiles) for smiles in df['canonical_smiles']]
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_descriptors = calc.pandas(mols)
    df_mordred = pd.concat([df, mordred_descriptors], axis=1)
    df_mordred = df_mordred.dropna(axis=1, how='all')
    return df_mordred

def calculate_ecfp(df, radius, prefix):
    mols = [Chem.MolFromSmiles(smiles) for smiles in df['canonical_smiles']]
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mol) for mol in mols]
    vectors = np.array([list(map(int, fp.ToBitString())) for fp in fingerprints])
    ecfp_df = pd.DataFrame(vectors)
    ecfp_df.columns = [f"{prefix}_{i}" for i in range(2048)]
    return pd.concat([df.reset_index(drop=True), ecfp_df], axis=1)

def evaluate_model(y_true, y_pred, label):
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"{label} Selectivity Metrics:\nAUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, MCC: {mcc:.4f}")

def main(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    # Load models and features
    xgb_model = joblib.load('../../models/classification/activity/xgb_final_pipeline_s1r.joblib')
    et_model = joblib.load('../../models/classification/activity/et_final_pipeline_s2r.joblib')
    svm_s1r_model = joblib.load('../../models/classification/selectivity/svm_final_pipeline_s1r.joblib')
    svm_s2r_model = joblib.load('../../models/classification/selectivity/svm_final_pipeline_s2r.joblib')

    selected_features_activity_s1r = joblib.load('../../models/classification/activity/selected_features_s1r.joblib')
    selected_features_activity_s2r = joblib.load('../../models/classification/activity/selected_features_s2r.joblib')
    selected_features_selectivity_s1r = joblib.load('../../models/classification/selectivity/selected_features_s1r.joblib')
    selected_features_selectivity_s2r = joblib.load('../../models/classification/selectivity/selected_features_s2r.joblib')

    # Descriptor Calculation
    df_mordred = sanitize_and_compute_descriptors(df)
    drop_cols = ['activity S1R', 'activity S2R', 'selectivity', 'pActivity S1R', 'pActivity S2R',
                 'Standard Relation S1R', 'Standard Relation S2R', 'canonical_smiles',
                 'activity_labels_S1R', 'activity_labels_S2R', 'selectivity_labels_S1R', 'selectivity_labels_S2R']
    df_features = df_mordred.drop(columns=[c for c in drop_cols if c in df_mordred.columns])

    X_s1r = df_features[selected_features_activity_s1r]
    X_s2r = df_features[selected_features_activity_s2r]
    pred_s1r = xgb_model.predict(X_s1r)
    pred_s2r = et_model.predict(X_s2r)

    df['activity_classification'] = [
        'S1R Selective' if s1r == 1 and s2r == 0 else
        'S2R Selective' if s1r == 0 and s2r == 1 else
        'Inactive' if s1r == 0 and s2r == 0 else
        'Active for Both' for s1r, s2r in zip(pred_s1r, pred_s2r)
    ]

    df_active_both = df[df['activity_classification'] == 'Active for Both']

    # S1R Selectivity
    df_s1r = calculate_ecfp(df_active_both, radius=2, prefix='ecfp4')
    df_s1r.rename(columns={f'ecfp4_{i}': str(i) for i in range(2048)}, inplace=True)

    #Add this: create label column for S1R
    df_s1r['selectivity_labels_S1R'] = df_s1r['selectivity'].apply(lambda x: 1 if x == 'S1R selective' else 0)

    X_s1r_sel = df_s1r[selected_features_selectivity_s1r]
    y_s1r = df_s1r['selectivity_labels_S1R']
    pred_sel_s1r = svm_s1r_model.predict(X_s1r_sel)
    evaluate_model(y_s1r, pred_sel_s1r, label="S1R")

    print("\nClassification Report for S1R Selectivity:")
    print(classification_report(y_s1r, pred_sel_s1r, target_names=["Non-selective", "S1R Selective"]))



    # S2R Selectivity
    df_s2r = calculate_ecfp(df_active_both, radius=3, prefix='ecfp6')
    df_s2r.rename(columns={f'ecfp6_{i}': str(i) for i in range(2048)}, inplace=True)

    #Add this: create label column for S2R
    df_s2r['selectivity_labels_S2R'] = df_s2r['selectivity'].apply(lambda x: 1 if x == 'S2R selective' else 0)

    X_s2r_sel = df_s2r[selected_features_selectivity_s2r]
    y_s2r = df_s2r['selectivity_labels_S2R']
    pred_sel_s2r = svm_s2r_model.predict(X_s2r_sel)
    evaluate_model(y_s2r, pred_sel_s2r, label="S2R")


    print("\nClassification completed successfully.")
    print("\nClassification Report for S2R Selectivity:")
    print(classification_report(y_s2r, pred_sel_s2r, target_names=["Non-selective", "S2R Selective"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classification workflow for S1R/S2R activity and selectivity.")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Directory to save outputs')
    args = parser.parse_args()
    main(args.input, args.output)
