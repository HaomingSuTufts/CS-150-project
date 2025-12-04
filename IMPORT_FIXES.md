# Import Fixes Summary

## Issue
The project had incorrect import statements using absolute imports instead of relative imports within the `gdss` package.

## Files Fixed

### 1. `src/gdss/core/losses.py`
- **Before:** `from .utils.graph_utils import ...`
- **After:** `from ..utils.graph_utils import ...`
- **Reason:** `utils` is a sibling package to `core`, not a child package

### 2. `src/gdss/core/solver_guidance.py`
- **Before:**
  ```python
  from losses import get_score_fn
  from utils.graph_utils import mask_adjs, mask_x, gen_noise
  from sde import VPSDE, subVPSDE
  import losses_guidance
  ```
- **After:**
  ```python
  from .losses import get_score_fn
  from ..utils.graph_utils import mask_adjs, mask_x, gen_noise
  from .sde import VPSDE, subVPSDE
  from . import losses_guidance
  ```

### 3. `src/gdss/utils/loader.py`
- **Before:**
  ```python
  from models.ScoreNetwork_A import ScoreNetworkA
  from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
  from sde import VPSDE, VESDE, subVPSDE
  from losses import get_sde_loss_fn
  from utils.ema import ExponentialMovingAverage
  ```
- **After:**
  ```python
  from ..models.ScoreNetwork_A import ScoreNetworkA
  from ..models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
  from ..core.sde import VPSDE, VESDE, subVPSDE
  from ..core.losses import get_sde_loss_fn
  from .ema import ExponentialMovingAverage
  ```

### 4. `src/gdss/utils/data_loader.py`
- **Before:** `from utils.graph_utils import init_features, graphs_to_tensor`
- **After:** `from .graph_utils import init_features, graphs_to_tensor`

### 5. `src/gdss/models/ScoreNetwork_X.py`
- **Before:**
  ```python
  from models.layers import DenseGCNConv, MLP
  from utils.graph_utils import mask_x, pow_tensor
  from models.attention import AttentionLayer
  ```
- **After:**
  ```python
  from .layers import DenseGCNConv, MLP
  from ..utils.graph_utils import mask_x, pow_tensor
  from .attention import AttentionLayer
  ```

### 6. `src/gdss/models/ScoreNetwork_A.py`
- **Before:**
  ```python
  from models.layers import DenseGCNConv, MLP
  from utils.graph_utils import mask_adjs, pow_tensor
  from models.attention import AttentionLayer
  ```
- **After:**
  ```python
  from .layers import DenseGCNConv, MLP
  from ..utils.graph_utils import mask_adjs, pow_tensor
  from .attention import AttentionLayer
  ```

### 7. `src/gdss/models/attention.py`
- **Before:**
  ```python
  from models.layers import DenseGCNConv, MLP
  from utils.graph_utils import mask_adjs, mask_x
  ```
- **After:**
  ```python
  from .layers import DenseGCNConv, MLP
  from ..utils.graph_utils import mask_adjs, mask_x
  ```

## Import Path Reference

For files in the `gdss` package structure:
- `.module` = sibling module in same package
- `..module` = module in parent package
- `...module` = module in grandparent package

### Package Structure:
```
src/gdss/
├── core/         (losses.py, sde.py, solver_guidance.py, losses_guidance.py)
├── models/       (ScoreNetwork_A.py, ScoreNetwork_X.py, layers.py, attention.py)
├── utils/        (loader.py, data_loader.py, graph_utils.py, mol_utils.py, etc.)
└── parsers/      (config.py, parser.py)
```

## Remaining External Dependencies

Some imports could not be fixed as they refer to external packages or missing modules:
- `prodigy.project_bisection` - External package (needs installation)
- `evaluation.mmd` - Missing module (needs to be created or installed)
- `data.data_generators` - Missing module (needs to be created or installed)
- `moses.metrics` - External package (already installed)

To install missing external packages:
```bash
pip install prodigy
```

To create missing modules, create:
- `src/evaluation/` package
- `src/data/` package
