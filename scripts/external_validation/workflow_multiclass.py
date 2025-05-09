import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.metrics import (classification_report, accuracy_score, 
                             f1_score, precision_score, recall_score, 
                             matthews_corrcoef, confusion_matrix)

def calculate_descriptors(df):
    mols = [Chem.MolFromSmiles(smiles) for smiles in df['canonical_smiles']]
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_descriptors = calc.pandas(mols)
    return pd.concat([df, mordred_descriptors], axis=1)

def define_multiclass_label(row, target):
    if row['selectivity'] == f'{target} selective' and row['activity'] == 'active':
        return 1
    elif row['activity'] == 'inactive':
        return 0
    else:
        return 2

def evaluate_predictions(true_labels, predictions, target, output_dir):
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)
    mcc = matthews_corrcoef(true_labels, predictions)

    report_df = pd.DataFrame(report).transpose()
    report_df['support'] = report_df['support'].astype(int)
    report_df.loc['MCC'] = {'precision': '', 'recall': '', 'f1-score': '', 'support': '', 'MCC': mcc}
    report_df.to_csv(os.path.join(output_dir, f'classification_metrics_{target}.csv'))

    print(f"\nClassification Report for {target} with MCC:")
    print(report_df)

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Inactive', 'Selective', 'Non-selective'],
                yticklabels=['Inactive', 'Selective', 'Non-selective'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {target}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{target}.png'))
    plt.close()

def run_model_prediction(df, target, model_path, features_path, output_dir):
    df_target = df[df['Target Name'] == target].reset_index(drop=True)
    df_target[f'{target}_multiclass_label'] = df_target.apply(lambda row: define_multiclass_label(row, target), axis=1)

    df_desc = calculate_descriptors(df_target)

    drop_cols = [
        'ID', 'activity', 'selectivity', 'pActivity', 'Target Name', 'Target Organism',
        'Standard Relation', 'canonical_smiles', 'Standard Value', 'Standard Type',
        'Standard Units', 'original_source', f'{target}_multiclass_label'
    ]
    df_features = df_desc.drop(columns=[col for col in drop_cols if col in df_desc.columns])

    model = joblib.load(model_path)
    selected_features = joblib.load(features_path)
    X = df_features[selected_features]

    predictions = model.predict(X)
    df_target['predictions'] = predictions

    evaluate_predictions(df_target[f'{target}_multiclass_label'], predictions, target, output_dir)
    print(f"\nFinished predictions for {target}. Results saved to {output_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="Run multiclass classification for S1R and S2R.")
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Directory to save outputs')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.input)

    run_model_prediction(
        df, 'S1R',
        model_path='../../models/multiclassification/et_final_pipeline_s1r.joblib',
        features_path='../../models/multiclassification/selected_features_s1r.joblib',
        output_dir=args.output
    )

    run_model_prediction(
        df, 'S2R',
        model_path='../../models/multiclassification/et_final_pipeline_s2r.joblib',
        features_path='../../models/multiclassification/selected_features_s2r.joblib',
        output_dir=args.output
    )

if __name__ == '__main__':
    main()