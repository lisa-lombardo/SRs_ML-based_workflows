#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import argparse

logging.basicConfig(level=logging.INFO)

def clean_chembl(file_path, target_name):
    df = pd.read_csv(file_path, sep=";")
    df.replace({"'='": '=', "'>'": '>'}, inplace=True)
    df.replace({"Kd1": "Kd", "Kd2": "Kd"}, inplace=True)
    df = df[df['Standard Type'].isin(['IC50', 'Ki', 'Kd'])]
    df.dropna(subset=['Smiles'], inplace=True)
    df = df[df['Standard Relation'].isin(['=', '>'])]
    df['Target Name'] = df['Target Name'].map({
    'Sigma opioid receptor': 'S1R',
    'Sigma intracellular receptor 2': 'S2R'
    })
    df.rename(columns={'pChEMBL Value': 'pActivity',
                       'Molecule ChEMBL ID': 'ID'}, inplace=True)
    df['original_source'] = 'ChEMBL'
    
    return df

def clean_bindingdb(file_path, target_name):
    df = pd.read_csv(file_path)
    df.rename(columns={
        'Ligand SMILES': 'Smiles',
        'Target Source Organism According to Curator or DataSource': 'Target Organism',
        'BindingDB Ligand Name': 'ID'
    }, inplace=True)
    df['Target Organism'] = 'Homo Sapiens'
    df['Target Name'] = df['Target Name'].map({
    'Sigma non-opioid intracellular receptor 1': 'S1R',
    'Sigma intracellular receptor 2': 'S2R'
    })
    df['Standard Value'] = np.nan
    df['Standard Type'] = np.nan
    df['Standard Units'] = np.nan

    def update_std(row):
        for col in ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']:
            if pd.notna(row.get(col)):
                row['Standard Value'] = row[col]
                row['Standard Type'] = col.split(' ')[0]
                row['Standard Units'] = col.split(' ')[1].strip('()')
                break
        return row

    df = df.apply(update_std, axis=1)

    def update_relation(row):
        val = row['Standard Value']
        if isinstance(val, str) and val[0] in ['<', '>']:
            row['Standard Relation'] = val[0]
            row['Standard Value'] = float(val[1:])
        else:
            try:
                float(val)
                row['Standard Relation'] = '='
            except:
                pass
        return row

    df['Standard Relation'] = np.nan
    df = df.apply(update_relation, axis=1)
    df = df[df['Standard Relation'].isin(['=', '>'])]
    df['original_source'] = 'BindingDB'
    return df

def clean_pubchem(file_path, target_name):
    df = pd.read_csv(file_path, sep=";")
    receptor_code = 'Q99720' if target_name == 'S1R' else 'Q5BJF2'
    df = df[df['protacxn'] == receptor_code]
    df.rename(columns={
        'acname': 'Standard Type',
        'targetname': 'Target Name',
        'acqualifier': 'Standard Relation',
        'acvalue': 'Standard Value',
        'cid': 'ID'
    }, inplace=True)
    df['Target Name'] = target_name
    df = df[df['Standard Type'].isin(['IC50', 'Ki', 'Kd'])]

    def convert_to_nM(value):
        try:
            return float(value) * 1000 if float(value) > 0 else np.nan
        except:
            return np.nan

    df['Standard Value'] = df['Standard Value'].apply(convert_to_nM)
    df['Standard Units'] = df['Standard Value'].apply(lambda x: 'nM' if pd.notna(x) and x > 0 else np.nan)
    df = df[df['Standard Relation'].isin(['=', '>'])]
    df['original_source'] = 'PubChem'
    return df

def clean_s2rsldb(file_path, target_name):
    df = pd.read_csv(file_path, sep=";")
    df.columns = df.columns.str.strip() 

    df = df[[
        "SMILES", "ID",
        f"Relation {target_name} Receptor",
        f"{target_name} Binding Affinity (nM)",
        f"Standard Type {target_name} Receptor"
    ]]

    df.columns = ['Smiles', 'ID', 'Standard Relation', 'Standard Value', 'Standard Type']
    df['Target Name'] = target_name
    df['Target Organism'] = 'Homo sapiens'
    df['Standard Units'] = 'nM'
    df = df[df['Standard Type'].isin(['IC50', 'Ki'])]
    df['original_source'] = 'S2RSLDB'
    return df

def clean_inhouse(file_path, target_name):
    df = pd.read_csv(file_path, sep=';')
    df.rename(columns={'Column1': 'ID'}, inplace=True)
    df.replace({"='": '=', "'='": '=', "'>'": '>'}, inplace=True)
    df['Target Name'] = target_name
    df['original_source'] = 'in-house'
    return df

def canonicalize_and_filter(df):
    def canonical(smiles):
        try:
            return Chem.CanonSmiles(smiles)
        except:
            return np.nan
    df['canonical_smiles'] = df['Smiles'].apply(canonical)
    df.dropna(subset=['canonical_smiles'], inplace=True)

    remover = SaltRemover(defnData="[Cl,Br]")

    def is_salt(smiles):
        mol = Chem.MolFromSmiles(smiles)
        stripped = remover.StripMol(mol) if mol else None
        return mol and stripped and mol.GetNumAtoms() != stripped.GetNumAtoms()

    df = df[~df['canonical_smiles'].apply(is_salt)]
    return df

def deduplicate_and_finalize(df):
    df = df.copy()
    df['Standard Value'] = df['Standard Value'].astype(float)

    max_value = df.groupby('canonical_smiles')['Standard Value'].transform(lambda x: x.max())
    df = df[(df['Standard Value'] == max_value) | (df['Standard Relation'] != '>')]
    
    df['Standard Value'] = df['Standard Value'].astype(float)
    
    grouped = df.groupby('canonical_smiles').agg({
        'ID': 'first',
        'Target Name': 'first',
        'Target Organism': 'first',
        'Standard Relation': list,
        'Standard Value': list,
        'Standard Type': list,
        'Standard Units': 'first',
        'original_source': 'first'
    }).reset_index()

    def filter_relation(row):
        if '=' in row['Standard Relation'] and '>' in row['Standard Relation']:
            keep = [i for i, v in enumerate(row['Standard Relation']) if v == '=']
            row['Standard Value'] = [row['Standard Value'][i] for i in keep]
            row['Standard Type'] = [row['Standard Type'][i] for i in keep]
            row['Standard Relation'] = '='
        else:
            row['Standard Relation'] = row['Standard Relation'][0]
        return row

    grouped = grouped.apply(filter_relation, axis=1)

    priority = {'Ki': 1, 'IC50': 2, 'Kd': 3}

    def prioritize_type(row):
        if isinstance(row['Standard Type'], list) and len(set(row['Standard Type'])) > 1:
            sorted_indices = sorted(range(len(row['Standard Type'])), key=lambda i: priority.get(row['Standard Type'][i], 99))
            row['Standard Value'] = row['Standard Value'][sorted_indices[0]]
            row['Standard Type'] = row['Standard Type'][sorted_indices[0]]
        elif isinstance(row['Standard Value'], list):
            row['Standard Value'] = row['Standard Value'][0]
            row['Standard Type'] = row['Standard Type'][0]
        return row

    grouped = grouped.apply(prioritize_type, axis=1)

    discarded = []

    def filter_by_std(row):
        if isinstance(row['Standard Value'], list) and len(row['Standard Value']) > 1:
            mean_val = np.mean(row['Standard Value'])
            std_val = np.std(row['Standard Value'])
            if std_val / mean_val < 0.20:
                row['Standard Value'] = mean_val
            else:
                discarded.append(row)
                return pd.Series()
        return row

    grouped = grouped.apply(filter_by_std, axis=1).dropna(how='all').reset_index(drop=True)

    grouped = grouped[~((grouped['Standard Relation'] == '>') & (grouped['Standard Value'] < 10000))]

    def calc_pActivity(value):
        try:
            val = float(value)
            return -np.log10(val * 1e-9) if val > 0 else np.nan
        except:
            return np.nan

    grouped['pActivity'] = grouped['Standard Value'].apply(calc_pActivity)

    def classify_activity(pAct):
        if pd.isna(pAct):
            return np.nan
        return 'active' if pAct > 5 else 'inactive'

    grouped['activity'] = grouped['pActivity'].apply(classify_activity)
    return grouped

def main():
    parser = argparse.ArgumentParser(description="Clean and merge sigma receptor datasets.")
    parser.add_argument("--receptor", choices=["S1R", "S2R"], required=True, help="Choose receptor: S1R or S2R")
    parser.add_argument("--input_dir", type=str, default="../datasets_CSV/in", help="Input directory path")
    parser.add_argument("--output_dir", type=str, default="../datasets_CSV/out/final_datasets", help="Output directory path")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    receptor = args.receptor
    dfs = []

    dfs.append(clean_chembl(input_dir / f"{receptor}_dataset_ChEMBL.csv", receptor))
    dfs.append(clean_bindingdb(input_dir / f"{receptor}_dataset_BindingDB.csv", receptor))
    dfs.append(clean_pubchem(input_dir / f"{receptor}_dataset_pubchem.csv", receptor))
    dfs.append(clean_s2rsldb(input_dir / "S2RSLDB.csv", receptor))
    dfs.append(clean_inhouse(input_dir / f"in-house_data_{receptor}.csv", receptor))

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = canonicalize_and_filter(combined_df)
    final_df = deduplicate_and_finalize(combined_df)

    final_df.to_csv(output_dir / f"{receptor}_dataset_out.csv", index=False)
    logging.info(f"{receptor} dataset saved to {output_dir}")

if __name__ == "__main__":
    main()