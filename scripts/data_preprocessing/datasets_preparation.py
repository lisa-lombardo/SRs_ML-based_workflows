#!/usr/bin/env python
# coding: utf-8

"""
Title: Sigma Receptor Dataset Labeling and Splitting
Description: Generates labeled datasets for ML tasks and creates an external validation set.
Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def assign_selectivity(df):
    df['selectivity'] = None
    grouped = df.groupby('canonical_smiles')

    for _, group in grouped:
        if len(group) == 1:
            df.loc[group.index, 'selectivity'] = 'single_point'
        else:
            p1 = group.loc[group['Target Name'] == 'S1R', 'pActivity']
            p2 = group.loc[group['Target Name'] == 'S2R', 'pActivity']

            # Handle incomplete profiles safely
            if p1.empty or p2.empty:
                df.loc[group.index, 'selectivity'] = 'single_point'
                continue

            pActivity_S1R = p1.values[0]
            pActivity_S2R = p2.values[0]

            # Label logic
            if pActivity_S1R <= 5 and pActivity_S2R > 5:
                label = 'S2R selective'
            elif pActivity_S2R <= 5 and pActivity_S1R > 5:
                label = 'S1R selective'
            elif pActivity_S1R <= 5 and pActivity_S2R <= 5:
                label = 'non-binder'
            elif pActivity_S1R - pActivity_S2R >= 2:
                label = 'S1R selective'
            elif pActivity_S2R - pActivity_S1R >= 2:
                label = 'S2R selective'
            elif 1 < abs(pActivity_S1R - pActivity_S2R) < 2:
                label = 'semi-selective'
            else:
                label = 'dual binder'

            df.loc[group.index, 'selectivity'] = label

    return df

def create_datasets(df, output_dir):
    # Step 1: Identify double-point compounds
    df_double_points = df[df['selectivity'] != 'single_point'].reset_index(drop=True)

    external_dataset = pd.DataFrame()
    remaining_dataset = pd.DataFrame()

    # Step 2: Split 10% of double-point compounds into external dataset
    unique_smiles = df_double_points['canonical_smiles'].unique()
    ext_smiles, rem_smiles = train_test_split(unique_smiles, test_size=0.90, random_state=15)

    external_dataset = df_double_points[df_double_points['canonical_smiles'].isin(ext_smiles)]
    remaining_dataset = df_double_points[df_double_points['canonical_smiles'].isin(rem_smiles)]


    # Step 3: Add single-point compounds to the remaining dataset
    single_points = df[df['selectivity'] == 'single_point']
    remaining_dataset = pd.concat([remaining_dataset, single_points], ignore_index=True)

    # Step 4: Save external dataset
    external_dataset.to_csv(output_dir / "external_validation_set.csv", index=False)
    print(f"External dataset saved: {len(external_dataset)} rows")

    # Dataset 1: S1R activity
    df1 = remaining_dataset[remaining_dataset['Target Name'] == 'S1R'].copy()
    df1['labels'] = df1['activity'].map(lambda x: 1 if x == 'active' else 0)
    df1.to_csv(output_dir / "dataset_1.csv", index=False)

    # Dataset 2: S2R activity
    df2 = remaining_dataset[remaining_dataset['Target Name'] == 'S2R'].copy()
    df2['labels'] = df2['activity'].map(lambda x: 1 if x == 'active' else 0)
    df2.to_csv(output_dir / "dataset_2.csv", index=False)

    # Dataset 3: S1R selectivity
    remaining_dataset_double_points = remaining_dataset[
    (remaining_dataset['selectivity'] != 'single_point') & 
    (remaining_dataset['selectivity'] != 'non-binder')
    ].reset_index(drop=True)

    df3 = remaining_dataset_double_points[remaining_dataset_double_points['Target Name'] == 'S1R'].copy()
    df3['labels'] = df3['selectivity'].map(lambda x: 1 if x == 'S1R selective' else 0)
    df3.to_csv(output_dir / "dataset_3.csv", index=False)

    # Dataset 4: S2R selectivity
    df4 = remaining_dataset_double_points[remaining_dataset_double_points['Target Name'] == 'S2R'].copy()
    df4['labels'] = df4['selectivity'].map(lambda x: 1 if x == 'S2R selective' else 0)
    df4.to_csv(output_dir / "dataset_4.csv", index=False)
    remaining_dataset[remaining_dataset['Target Name'] == 'S1R'].copy()


    # Dataset 5: S1R selective and inactive
    S1R_inactive = remaining_dataset[(remaining_dataset['Target Name'] == 'S1R') & (remaining_dataset['activity'] == 'inactive')] 
    df5 = pd.concat([df3, S1R_inactive], ignore_index=True)
    
    # Multiclass labeling logic
    def multiclass_label(row):
        if row['activity'] == 'inactive':
            return 0
        elif row['selectivity'] in ['S1R selective', 'S2R selective']:
            return 1
        else:
            return 2

    df5['labels'] = df5.apply(multiclass_label, axis=1)
    df5.to_csv(output_dir / "dataset_5.csv", index=False)


    # Dataset 6: S2R selective and inactive
    S2R_inactive = remaining_dataset[(remaining_dataset['Target Name'] == 'S2R') & (remaining_dataset['activity'] == 'inactive')]
    df6 = pd.concat([df4, S2R_inactive], ignore_index=True)
    df6['labels'] = df6.apply(multiclass_label, axis=1)
    df6.to_csv(output_dir / "dataset_6.csv", index=False)

    print("All datasets created and saved.")

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for ML modeling.")
    parser.add_argument("--input_s1r", type=str, required=True, help="Input path to cleaned S1R dataset")
    parser.add_argument("--input_s2r", type=str, required=True, help="Input path to cleaned S2R dataset")
    parser.add_argument("--output_dir", type=str, default="../datasets_CSV/out/ml_datasets", help="Output directory path")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_S1R = pd.read_csv(args.input_s1r, sep=";")
    df_S2R = pd.read_csv(args.input_s2r, sep=";")
    df = pd.concat([df_S1R, df_S2R], ignore_index=True)

    df = assign_selectivity(df)
    create_datasets(df, output_dir)

if __name__ == "__main__":
    main()
