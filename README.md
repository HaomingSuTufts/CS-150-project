# CS-150 Project: Diffusion on Graphs

## 📘 Overview
This repository contains our class project for CS150, focusing on **diffusion processes on graphs**.  
We aim to explore how information, signals, or particles diffuse across network structures, and how these processes can be leveraged for **graph reconstruction** and related tasks.

## 🎯 Objectives
- Implement algorithms to simulate diffusion on graphs.
- Study reconstruction methods that infer graph structure from diffusion data.
- Provide reproducible experiments and clear documentation.

## 🛠️ Project Structure
- `src/` → Core implementation (Python scripts, Jupyter notebooks).
- `data/` → Example datasets and synthetic graph generators.
- `experiments/` → Scripts and configs for running experiments.
- `docs/` → Additional notes, references, and reports.

## 📦 Requirements
- Python 3.13
- Common libraries: `numpy`, `networkx`, `matplotlib`, `scipy`, 'pytorch'

Install dependencies:
```bash
pip install -r requirements.txt


## Generation (sampling)

Use `generation.py` to load a trained checkpoint and generate new graphs.

Example (CPU):

	python generation.py --dataset QM9 --ckpt gdss_qm9_training_best_epoch_802 --n_samples 16 --device cpu --seed 42

This will write generated graphs and SMILES (if applicable) to `samples/<DATASET>/<CKPT>/`.
