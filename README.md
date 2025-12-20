# Perspective to Equirectangular (P2E) ComfyUI Node

A custom ComfyUI node that projects perspective images onto an equirectangular panorama with optional feathered blending. The core implementation is written in PyTorch and now accepts either NumPy arrays or torch tensors, automatically adapting shapes and returning outputs in the same container type.

## Features
- Perspective-to-equirectangular projection with cached sampling grids for speed
- Feathered blending to soften seams
- Flexible input handling for NumPy arrays or torch tensors (HWC/CHW/BHWC/BCHW)
- Ready for use inside ComfyUI or as a standalone Python utility

## Installation
1. Clone or download this repository.
2. Copy the `p2e` folder into your `ComfyUI/custom_nodes/` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Restart ComfyUI.

## Node Usage
The node appears under `image/360` as **Perspective to Equirectangular + Blend**. Required inputs:
- **Perspective**: image tensor/array
- **Equirectangular Base**: target panorama
- **FOV W / FOV H**: horizontal and vertical field of view in degrees
- **U Deg / V Deg**: yaw and pitch rotations in degrees
- **Feather**: blur radius for edge blending

Outputs:
1. **Merged** panorama
2. **Patch 360**: warped perspective patch
3. **Mask**: blend mask (same container type as inputs)

## Standalone Usage
You can also call the core function directly:
```python
from p2e.nodes import p2e_and_blend_torch
import numpy as np

# perspective_np and equi_base_np are HWC NumPy arrays in [0, 255] or [0, 1]
merged, patch, mask = p2e_and_blend_torch(
    perspective=perspective_np,
    equi_base=equi_base_np,
    fov_deg=(140, 140),
    u_deg=180,
    v_deg=-70,
    feather=10,
)
```

Both NumPy arrays and torch tensors are supported; outputs mirror the input container type and layout.

## Notes
- If ComfyUI is available, the node will automatically use its configured torch device. Otherwise, it falls back to CUDA when available.
- Integer image inputs are normalized internally and restored to their original dtype on output.
