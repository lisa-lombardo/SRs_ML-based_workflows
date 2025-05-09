#!/usr/bin/env python
# coding: utf-8

"""
calculate_descriptors.py

Calculates molecular descriptors (RDKit, Mordred) and fingerprints (ECFP4, ECFP6, MACCS)
from a cleaned dataset of ligands. Outputs are saved as feature matrices.

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from mordred import Calculator, descriptors
from pathlib import Path
import argparse

COLUMNS_TO_DROP = [
    'canonical_smiles', 'mols', 'ID', 'Target Name', 'Target Organism',
    'Standard Relation', 'Standard Value', 'Standard Type', 'Standard Units', 'original_source'
]

def sanitize_and_convert(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"Sanitization error for SMILES: {smiles} - {e}")
        return None
    return mol

def filter_descriptors(df, variance_cutoff=0.1, correlation_cutoff=0.9):
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    df = df.loc[:, df.nunique() > 1]
    df = df.loc[:, df.var() > variance_cutoff]
    corr_matrix = df.corr().abs()
    to_drop = {corr_matrix.columns[i]
               for i in range(len(corr_matrix.columns))
               for j in range(i)
               if corr_matrix.iloc[i, j] > correlation_cutoff}
    return df.drop(columns=to_drop)

def calculate_rdkit_descriptors(df, output_dir):
    print("Calculating RDKit descriptors...")
    vals = df['mols'].apply(lambda x: Descriptors.CalcMolDescriptors(x))
    rdkit = pd.DataFrame(vals.tolist())
    rdkit_filtered = filter_descriptors(rdkit)
    df_rdkit = pd.concat([df, rdkit_filtered], axis=1).drop(columns=COLUMNS_TO_DROP)
    df_rdkit.to_csv(output_dir / "df_rdkit.csv", index=False)
    print(f"Saved RDKit descriptors: {df_rdkit.shape[1]} columns")


def calculate_mordred_descriptors(df, output_dir):
    print("Calculating Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=True)
    mordred = calc.pandas(df['mols'])
    mordred_filtered = filter_descriptors(mordred)
    df_mordred = pd.concat([df, mordred_filtered], axis=1).drop(columns=COLUMNS_TO_DROP)
    df_mordred.to_csv(output_dir / "df_mordred.csv", index=False)
    print(f"Saved Mordred descriptors: {df_mordred.shape[1]} columns")


def calculate_morgan_fingerprint(df, output_dir, radius, n_bits):
    print(f"Calculating Morgan fingerprints (radius={radius}, nBits={n_bits})...")
    generator = AllChem.GetMorganGenerator(radius=radius)
    fps = [generator.GetFingerprint(mol) for mol in df['mols']]
    bit_vectors = np.array([list(map(int, fp.ToBitString())) for fp in fps])
    df_fp = pd.DataFrame(bit_vectors)
    df_fp_full = pd.concat([df, df_fp], axis=1).drop(columns=COLUMNS_TO_DROP)

    if radius == 2:
        filename = "df_ECFP4.csv"
    elif radius == 3:
        filename = "df_ECFP6.csv"
    else:
        filename = f"df_morgan_r{radius}_{n_bits}bit.csv"

    df_fp_full.to_csv(output_dir / filename, index=False)
    print(f"Saved Morgan fingerprints: {df_fp.shape[1]} bits")


def calculate_maccs_keys(df, output_dir):
    print("Calculating MACCS fingerprints...")
    fps = [MACCSkeys.GenMACCSKeys(mol) for mol in df['mols']]
    bit_vectors = [list(map(int, fp.ToBitString())) for fp in fps]
    df_fp = pd.DataFrame(bit_vectors)
    df_fp_full = pd.concat([df, df_fp], axis=1).drop(columns=COLUMNS_TO_DROP)
    df_fp_full.to_csv(output_dir / "df_maccs.csv", index=False)
    print(f"Saved MACCS fingerprints: {df_fp.shape[1]} bits")


def main():
    parser = argparse.ArgumentParser(description="Calculate molecular descriptors and fingerprints.")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset (CSV)")
    parser.add_argument("--output_dir", type=str, default="../features", help="Base directory to save feature files")
    args = parser.parse_args()

    input_path = Path(args.input)
    dataset_name = input_path.stem  # Get filename without extension
    output_base = Path(args.output_dir)
    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input dataset: {input_path.name}")
    print(f"Output directory: {output_dir.resolve()}")

    df = pd.read_csv(input_path)
    df['mols'] = df['canonical_smiles'].apply(sanitize_and_convert)
    df = df[df['mols'].notna()].reset_index(drop=True)

    calculate_rdkit_descriptors(df, output_dir)
    calculate_mordred_descriptors(df, output_dir)
    calculate_morgan_fingerprint(df, output_dir, radius=2, n_bits=2048)
    calculate_morgan_fingerprint(df, output_dir, radius=3, n_bits=2048)
    calculate_maccs_keys(df, output_dir)

    print("\nAll descriptor and fingerprint files saved.")

if __name__ == "__main__":
    main()

