# data/ Directory

This folder contains all data assets used throughout the QSAR modeling project, including raw sources, cleaned datasets, molecular descriptors, and external validation data. Below is a description of each subdirectory and its purpose in the workflow.

---

##raw/
- **Purpose**: Store raw, unmodified datasets gathered from various sources (e.g., ChEMBL, PubChem, literature).
- **Format**: Original CSV/Excel files containing SMILES, bioactivity values, and metadata.
- **Note**: No processing or cleaning is done at this stage.

---

##processed/
- **Purpose**: Hold cleaned and standardized versions of raw datasets.
- **Includes**:
  - Canonicalized SMILES
  - Unified bioactivity columns (e.g., IC50, Ki)
  - Removed duplicates or inconsistent entries
- **Used by**: Descriptor/fingerprint calculation scripts.

---

##features/
- **Purpose**: Contains final feature matrices used for model training and evaluation.
- **Includes**:
  - Molecular descriptors (e.g., RDKit, Mordred)
  - Fingerprints (e.g., ECFP4, MACCS)
  - Target labels for classification, regression, or multiclass modeling
- **Structure**:
