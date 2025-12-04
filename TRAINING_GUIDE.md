# Training GDSS Models from Scratch

This guide explains how to train the Graph Diffusion via the System of SDEs (GDSS) model from scratch.

## Prerequisites

1. **Python Environment**: Python 3.12+ with required packages installed
2. **GPU**: CUDA-capable GPU recommended (8GB+ VRAM)
3. **Data**: QM9 or other graph datasets

## Quick Start

### 1. Install Dependencies

```bash
uv sync
# or
pip install -r requirements.txt
```

### 2. Prepare Data

The data should be automatically downloaded when you run the training script. Make sure the `data/` directory exists:

```bash
mkdir -p data
```

### 3. Train the Model

```bash
python train.py --config train_qm9 --seed 42
```

This will:
- Load the QM9 dataset
- Initialize the score networks
- Train for 3000 epochs (default)
- Save checkpoints every 100 epochs to `checkpoints/QM9/`

## Configuration

Training configurations are stored in `config/train_qm9.yaml`. Key parameters:

```yaml
train:
  num_epochs: 3000      # Total training epochs
  lr: 0.001             # Learning rate
  batch_size: 1024      # Batch size
  ema: 0.999            # EMA decay for stable sampling
```

### Adjusting for Different Datasets

For **ZINC250k**:
```yaml
data:
  data: ZINC250k
  max_node_num: 38
  max_feat_num: 4
```

For **custom datasets**, you'll need to:
1. Add data loading code in `src/gdss/utils/data_loader.py`
2. Create appropriate config files
3. Adjust `max_node_num` and `max_feat_num`

## Training Time

Expected training times on different hardware:

| GPU      | Batch Size | Time per Epoch | Total Time (3000 epochs)      |
| -------- | ---------- | -------------- | ----------------------------- |
| RTX 3090 | 1024       | ~2 min         | ~100 hours                    |
| RTX 4090 | 1024       | ~1.5 min       | ~75 hours                     |
| V100     | 512        | ~3 min         | ~150 hours                    |
| CPU      | 128        | ~30 min        | ~1500 hours (not recommended) |

**Tip**: You can start with fewer epochs (e.g., 1000) for initial testing.

## Checkpoints

Checkpoints are saved in `checkpoints/{DATASET_NAME}/` with the following naming:

- `{name}_epoch_100.pth`, `{name}_epoch_200.pth`, ... - Periodic checkpoints
- `{name}_best.pth` - Best model (lowest loss)
- `{name}_final.pth` - Final model after all epochs

### Checkpoint Structure

Each checkpoint contains:
```python
{
    'epoch': int,
    'config': dict,
    'params_x': dict,           # Model X hyperparameters
    'params_adj': dict,         # Model Adj hyperparameters
    'x_state_dict': dict,       # Model X weights
    'adj_state_dict': dict,     # Model Adj weights
    'optimizer_x': dict,        # Optimizer X state
    'optimizer_adj': dict,      # Optimizer Adj state
    'ema_x': dict,              # EMA for Model X
    'ema_adj': dict,            # EMA for Model Adj
}
```

## Using Trained Models in the Notebook

Once training is complete, update your sampling config (`config/sample_qm9.yaml`):

```yaml
# Point to your trained checkpoint
ckpt: gdss_qm9_training_final  # or gdss_qm9_training_best

# Use EMA weights for better sample quality
sample:
  use_ema: True
```

Then in the notebook (cell 5):
```python
config_file = 'sample_qm9'
config = get_config(config_file, seed)
```

## Monitoring Training

Training logs are saved to `logs_train/{dataset}/{experiment_name}/train.log`

You can monitor training progress:
```bash
tail -f logs_train/QM9/gdss_qm9_training/train.log
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `config/train_qm9.yaml`:
```yaml
data:
  batch_size: 512  # or 256
```

### Slow Training

- **Use GPU**: Make sure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- **Reduce model size**: Lower `nhid`, `depth`, or `num_heads` in config
- **Use mixed precision**: Add AMP (requires code modification)

### NaN Losses

- **Reduce learning rate**: Try `lr: 0.0001`
- **Enable gradient clipping**: Ensure `grad_clip: 1.0` is set
- **Check data**: Ensure no NaN/Inf values in dataset

### Data Not Found

If you get data loading errors:
1. Check that `data/` directory exists
2. Ensure dataset files are in correct format
3. For QM9: The loader should auto-download, but you may need to manually download from:
   - https://www.research-collection.ethz.ch/handle/20.500.11850/214853

## Alternative: Using Pre-trained Checkpoints

If you don't want to train from scratch, you can:

1. **Download from original authors**: Check the GDSS paper repository
   - Paper: https://arxiv.org/abs/2209.14734
   - Code: https://github.com/harryjo97/GDSS

2. **Contact authors**: Email the corresponding author for pre-trained weights

3. **Use transfer learning**: Start from a checkpoint trained on similar data

## Advanced Options

### Custom Model Architecture

Edit `config/train_qm9.yaml` to modify model architecture:

```yaml
model:
  x:
    depth: 5          # Deeper model (more layers)
    nhid: 256         # Wider model (more hidden units)
    num_heads: 8      # More attention heads
```

### Different SDE Types

Try different SDE formulations:

```yaml
sde:
  x:
    type: VESDE       # Variance Exploding SDE
    sigma_min: 0.01
    sigma_max: 50
  adj:
    type: VPSDE       # Variance Preserving SDE
```

### Learning Rate Scheduling

Adjust LR schedule:

```yaml
train:
  lr: 0.002
  lr_schedule: True
  lr_decay: 0.9995   # More aggressive decay
```

## Questions?

Refer to:
- Original GDSS paper: https://arxiv.org/abs/2209.14734
- Project issues: Create an issue in the repository
- The training logs for detailed debugging information
