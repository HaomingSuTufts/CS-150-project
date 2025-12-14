import pickle
import numpy as np
import os
import sys
import rdkit
import rdkit.Chem as Chem
import argparse

# Add src to path so we can import internal modules as packages from root
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gdss.utils.mol_utils import gen_mol
import torch


parser = argparse.ArgumentParser()
parser.add_argument(
    "--adj_path", type=str, required=True, help="Path to pickled adjacency matrices"
)
parser.add_argument(
    "--node_path", type=str, required=True, help="Path to pickled node matrices"
)
parser.add_argument(
    "--save_path",
    type=str,
    required=False,
    help="Path to save generated molecules smiles",
)
args = parser.parse_args()

node_matrices = np.load(args.node_path)  # (10000, 9, 5)
adj_matrices = np.load(args.adj_path)  # (10000, 9, 9)

mols = []
for i in range(node_matrices.shape[0]):
    node_tensor = torch.tensor(node_matrices[i : i + 1], dtype=torch.float32)  # (1,9,5)

    # keep batch dim and convert integer adjacency (1,9,9) -> one-hot (1,4,9,9)
    adj_tensor = torch.tensor(adj_matrices[i : i + 1], dtype=torch.long)  # (1,9,9)
    adj_onehot = torch.nn.functional.one_hot(adj_tensor, num_classes=4)  # (1,9,9,4)
    adj_onehot = adj_onehot.permute(0, 3, 1, 2).float()  # (1,4,9,9)

    gened_mols, _, _ = gen_mol(
        node_tensor, adj_onehot, dataset="QM9", largest_connected_comp=True
    )
    mols.extend(gened_mols)

smiles_list = [Chem.MolToSmiles(mol) if mol is not None else "" for mol in mols]
# Save or print SMILES
if args.save_path:
    with open(f"{args.save_path}/smiles.txt", "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")
else:
    for smi in smiles_list:
        print(smi)
