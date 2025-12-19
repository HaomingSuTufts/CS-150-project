"""
Script to prepare qm9_kekulized.npz from a QM9 CSV file with SMILES

Usage:
    python scripts/prepare_qm9_kekulized.py --csv data/qm9.csv --out data/qm9_kekulized.npz --test-split 0.1

This script will:
- Read SMILES from CSV (column 'smiles' for QM9)
- Kekulize and canonicalize with RDKit
- Build atom number arrays (padded to out_size) and 4-edge-type adjacency arrays
- Save them as arr_0 and arr_1 in NPZ
- Create valid_idx_qm9.json with randomly selected test indices if not present

"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from gdss.utils.smile_to_graph import GGNNPreprocessor

# RDKit import as used elsewhere
try:
    from rdkit import Chem as RDKitChem
except Exception:
    raise ImportError(
        "RDKit not available in the Python environment. Please install RDKit and try again."
    )


def prepare_qm9(
    csv_path,
    out_path,
    max_atoms=9,
    out_size=9,
    kekulize=True,
    test_split=0.1,
    valid_idx_path=None,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Please place the file at the path or pass a valid CSV path using --csv"
        )

    df = pd.read_csv(csv_path)
    # QM9 uses column name 'SMILES1'
    if (
        "SMILES1" not in df.columns
        and "smiles" not in df.columns
        and "SMILES" not in df.columns
    ):
        raise ValueError(
            "CSV does not contain a recognized SMILES column (SMILES1 / SMILES / smiles) for QM9 dataset"
        )
    # Try to find a suitable column name
    if "SMILES1" in df.columns:
        smiles_col = "SMILES1"
    elif "SMILES" in df.columns:
        smiles_col = "SMILES"
    else:
        smiles_col = "smiles"

    smiles_list = df[smiles_col].tolist()
    print(f"Found {len(smiles_list)} SMILES in CSV")

    preproc = GGNNPreprocessor(
        max_atoms=max_atoms, out_size=out_size, kekulize=kekulize
    )

    atom_arrays = []
    adj_arrays = []
    valid_indices = []

    for i, smi in enumerate(tqdm(smiles_list, desc="Processing SMILES")):
        if not isinstance(smi, str) or len(smi.strip()) == 0:
            continue
        try:
            mol = RDKitChem.MolFromSmiles(smi)
            if mol is None:
                continue
            # prepare and kekulize as in preprocessor
            smi_can, mol2 = preproc.prepare_smiles_and_mol(mol)
            atom_array, adj_array = preproc.get_input_features(mol2)
            atom_arrays.append(atom_array)
            adj_arrays.append(adj_array)
            valid_indices.append(i)
        except Exception as e:
            # ignore failures
            print(f"Failed to process index {i} with SMILES {smi}: {e}")
            continue

    if len(atom_arrays) == 0:
        raise RuntimeError("No molecules were processed successfully")

    arr_0 = np.stack(atom_arrays)  # (N, out_size)
    arr_1 = np.stack(adj_arrays)  # (N, 4, out_size, out_size)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, arr_0=arr_0, arr_1=arr_1)
    print(f"Saved NPZ to {out_path} with {arr_0.shape[0]} molecules")

    # Create valid_idx json if not present
    if valid_idx_path is None:
        valid_idx_path = os.path.join(os.path.dirname(out_path), "valid_idx_qm9.json")

    if not os.path.exists(valid_idx_path):
        num_test = max(1, int(len(atom_arrays) * test_split))
        np.random.seed(42)
        test_idxs = np.random.choice(
            len(atom_arrays), size=num_test, replace=False
        ).tolist()
        valid_idx_dict = {"valid_idxs": test_idxs}
        with open(valid_idx_path, "w") as f:
            json.dump(valid_idx_dict, f)
        print(
            f"Created valid_idx JSON with {num_test} test indices at {valid_idx_path}"
        )
    else:
        print(f"valid_idx JSON already exists at {valid_idx_path}; not overwriting it")

    return out_path, valid_idx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare QM9 kekulized NPZ file")
    parser.add_argument(
        "--csv", type=str, default="data/qm9.csv", help="Path to QM9 CSV file"
    )
    parser.add_argument(
        "--out", type=str, default="data/qm9_kekulized.npz", help="Output NPZ path"
    )
    parser.add_argument("--max_atoms", type=int, default=9, help="Maximum atoms")
    parser.add_argument("--out_size", type=int, default=9, help="Out size (padding)")
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction for test split to create valid_idx JSON",
    )
    parser.add_argument(
        "--valid-idx",
        type=str,
        default=None,
        help="Path to save valid_idx JSON (optional)",
    )

    args = parser.parse_args()
    prepare_qm9(
        args.csv,
        args.out,
        args.max_atoms,
        args.out_size,
        True,
        args.test_split,
        args.valid_idx,
    )
