"""
ComfyUI Node: Perspective to Equirectangular with Blending

Pure PyTorch implementation for projecting perspective images onto equirectangular
panoramas with seamless blending.
"""

import torch
import torch.nn.functional as F
import numpy as np

try:
    import comfy.model_management as mm

    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False


def _to_bhwc_tensor(image, device=None):
    """Convert a NumPy array or torch tensor image to BHWC float32 tensor.

    Returns the normalized tensor along with metadata required to restore the
    original layout, dtype, and container type.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image, np.ndarray):
        input_type = "numpy"
        orig_dtype = image.dtype
        tensor = torch.from_numpy(image)
        orig_device = device
    elif torch.is_tensor(image):
        input_type = "torch"
        orig_dtype = image.dtype
        tensor = image
        orig_device = image.device
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected numpy array or torch tensor."
        )

    layout = None
    if tensor.dim() == 2:
        # H x W → 1-channel BHWC
        tensor = tensor.unsqueeze(-1).unsqueeze(0)
        layout = "hw"
    elif tensor.dim() == 3:
        if tensor.shape[-1] in (1, 3, 4):
            # H x W x C
            tensor = tensor.unsqueeze(0)
            layout = "hwc"
        elif tensor.shape[0] in (1, 3, 4):
            # C x H x W
            tensor = tensor.permute(1, 2, 0).unsqueeze(0)
            layout = "chw"
        else:
            raise ValueError(f"Unsupported 3D layout with shape {tuple(tensor.shape)}")
    elif tensor.dim() == 4:
        if tensor.shape[-1] in (1, 3, 4):
            layout = "bhwc"
        elif tensor.shape[1] in (1, 3, 4):
            tensor = tensor.permute(0, 2, 3, 1)
            layout = "bchw"
        else:
            raise ValueError(f"Unsupported 4D layout with shape {tuple(tensor.shape)}")
    else:
        raise ValueError(
            f"Unsupported tensor dimensions: {tensor.dim()}. Expected 2D-4D image tensor."
        )

    # Normalize to float32 in [0,1]
    if tensor.is_floating_point():
        norm = tensor.to(device=device, dtype=torch.float32)
        scale = None
    else:
        info = torch.iinfo(tensor.dtype)
        norm = tensor.to(device=device, dtype=torch.float32) / float(info.max)
        scale = float(info.max)

    metadata = {
        "input_type": input_type,
        "orig_dtype": orig_dtype,
        "orig_device": orig_device,
        "layout": layout,
        "scale": scale,
    }

    return norm, metadata


def _restore_from_bhwc(tensor, metadata):
    """Restore tensor to original container type, layout, dtype, and device."""

    layout = metadata["layout"]
    if layout == "hw":
        tensor = tensor.squeeze(0).squeeze(-1)
    elif layout == "hwc":
        tensor = tensor.squeeze(0)
    elif layout == "chw":
        tensor = tensor.squeeze(0).permute(2, 0, 1)
    elif layout == "bhwc":
        pass
    elif layout == "bchw":
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unknown layout metadata: {layout}")

    # Rescale if original data was integer
    if metadata["scale"]:
        tensor = (tensor * metadata["scale"]).round().clamp(0, metadata["scale"])

    # Cast and place back on original container
    if metadata["input_type"] == "torch":
        tensor = tensor.to(device=metadata["orig_device"], dtype=metadata["orig_dtype"])
        return tensor

    # NumPy output
    tensor = tensor.detach().cpu()
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    array = tensor.numpy().astype(metadata["orig_dtype"], copy=False)
    return array


def p2e_and_blend_torch(
    perspective,
    equi_base,
    fov_deg: tuple,
    u_deg: float,
    v_deg: float,
    feather: int = 0,
    device: torch.device = None,
) -> tuple:
    """
    Project perspective image onto equirectangular base with blending.

    Args:
        perspective: torch.Tensor or np.ndarray - Perspective image(s)
        equi_base: torch.Tensor or np.ndarray - Equirectangular base image(s)
        fov_deg: tuple (fov_w, fov_h) - Field of view in degrees
        u_deg: float - Horizontal rotation in degrees (yaw)
        v_deg: float - Vertical rotation in degrees (pitch)
        feather: int - Feather radius for smooth blending (default: 0)
        device: torch.device - Device to use (default: auto-detect)

    Returns:
        tuple of (merged, patch_360, mask):
            - merged: BHWC, float32, [0,1] - Blended result
            - patch_360: BHWC, float32, [0,1] - Perspective warped to equirectangular
            - mask: BHW, float32, [0,1] - Blend mask
    """
    # Device management
    if device is None:
        if COMFY_AVAILABLE:
            device = mm.get_torch_device()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize and move inputs to device as BHWC float32
    perspective_t, persp_meta = _to_bhwc_tensor(perspective, device=device)
    equi_base_t, equi_meta = _to_bhwc_tensor(equi_base, device=device)

    if persp_meta["input_type"] != equi_meta["input_type"]:
        raise TypeError(
            "perspective and equi_base must be the same container type (both numpy or both torch)."
        )

    # Batch handling
    b_persp = perspective_t.shape[0]
    b_equi = equi_base_t.shape[0]

    if b_persp == b_equi:
        batch_size = b_persp
    elif b_persp == 1:
        batch_size = b_equi
        perspective_t = perspective_t.expand(b_equi, -1, -1, -1)
    elif b_equi == 1:
        batch_size = b_persp
        equi_base_t = equi_base_t.expand(b_persp, -1, -1, -1)
    else:
        raise ValueError(
            f"Incompatible batch sizes: perspective={b_persp}, equi_base={b_equi}. "
            f"Expected: both equal, or one equals 1."
        )

    # Initialize caches as function attributes
    if not hasattr(p2e_and_blend_torch, "_grid_cache"):
        p2e_and_blend_torch._grid_cache = {}
    if not hasattr(p2e_and_blend_torch, "_gauss_cache"):
        p2e_and_blend_torch._gauss_cache = {}

    # Extract dimensions
    h_eq, w_eq = equi_base_t.shape[1:3]
    h_p, w_p = perspective_t.shape[1:3]
    fov_w, fov_h = float(fov_deg[0]), float(fov_deg[1])

    # Create cache key with device string
    device_str = str(device)
    cache_key = (
        h_eq,
        w_eq,
        h_p,
        w_p,
        fov_w,
        fov_h,
        float(u_deg),
        float(v_deg),
        device_str,
    )

    # -------------------------
    # Grid generation (cached)
    # -------------------------
    if cache_key not in p2e_and_blend_torch._grid_cache:
        # Create equirectangular coordinate grid
        eq_y, eq_x = torch.meshgrid(
            torch.arange(h_eq, device=device, dtype=torch.float32),
            torch.arange(w_eq, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Convert to spherical coordinates
        theta = (eq_x / w_eq - 0.5) * (2.0 * torch.pi)
        phi = -(eq_y / h_eq - 0.5) * torch.pi

        # Convert to 3D unit vectors
        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi)
        z = torch.cos(phi) * torch.cos(theta)
        xyz = torch.stack([x, y, z], dim=-1)

        # Rotation angles
        u = torch.deg2rad(
            torch.tensor(float(u_deg), device=device, dtype=torch.float32)
        )
        v = torch.deg2rad(
            torch.tensor(float(-v_deg), device=device, dtype=torch.float32)
        )

        cu, su = torch.cos(u), torch.sin(u)
        cv, sv = torch.cos(v), torch.sin(v)

        z0 = torch.tensor(0.0, device=device, dtype=torch.float32)
        o1 = torch.tensor(1.0, device=device, dtype=torch.float32)

        # Rotation matrices
        Ry = torch.stack(
            [
                torch.stack([cu, z0, -su]),
                torch.stack([z0, o1, z0]),
                torch.stack([su, z0, cu]),
            ],
            dim=0,
        )

        Rx = torch.stack(
            [
                torch.stack([o1, z0, z0]),
                torch.stack([z0, cv, -sv]),
                torch.stack([z0, sv, cv]),
            ],
            dim=0,
        )

        # Apply rotation
        R = Ry @ Rx
        xyz_cam = xyz @ R

        # Check validity (in front of camera)
        valid = xyz_cam[..., 2] > 0
        z_safe = torch.where(valid, xyz_cam[..., 2], torch.ones_like(xyz_cam[..., 2]))

        # Project to perspective coordinates
        x_p = xyz_cam[..., 0] / z_safe
        y_p = xyz_cam[..., 1] / z_safe

        # FOV calculations
        tan_h = torch.tan(
            torch.deg2rad(torch.tensor(fov_w, device=device, dtype=torch.float32)) / 2.0
        )
        tan_v = torch.tan(
            torch.deg2rad(torch.tensor(fov_h, device=device, dtype=torch.float32)) / 2.0
        )

        # Check if within FOV
        in_fov = valid & (x_p.abs() <= tan_h) & (y_p.abs() <= tan_v)

        # Convert to pixel coordinates
        u_px = (x_p / tan_h + 1.0) * 0.5 * w_p
        v_px = (-y_p / tan_v + 1.0) * 0.5 * h_p

        # Create mapping coordinates
        map_x = torch.where(in_fov, u_px, torch.full_like(u_px, -1.0))
        map_y = torch.where(in_fov, v_px, torch.full_like(v_px, -1.0))

        # Normalize to [-1, 1] for grid_sample
        grid_x = (map_x / (w_p - 1.0)) * 2.0 - 1.0
        grid_y = (map_y / (h_p - 1.0)) * 2.0 - 1.0

        # Create grid [1, H, W, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Create mask (HW, uint8 with values 0 or 255)
        mask_t = in_fov.to(torch.uint8) * 255

        # Store in cache
        p2e_and_blend_torch._grid_cache[cache_key] = (grid, mask_t)
    else:
        grid, mask_t = p2e_and_blend_torch._grid_cache[cache_key]

    # -------------------------
    # Grid sampling
    # -------------------------
    # Ensure grid is float32 (in case old cache has different dtype)
    grid = grid.to(dtype=torch.float32)

    # Convert BHWC → BCHW for grid_sample
    pers_bchw = perspective_t.permute(0, 3, 1, 2)

    # Scale to [0,255] range for better float32 precision during interpolation
    # (matches original implementation which loaded uint8 images)
    pers_bchw = pers_bchw * 255.0

    # Expand grid for batch
    grid_batch = grid.expand(batch_size, -1, -1, -1)

    # Apply grid_sample (requires float32 inputs)
    out_bchw = F.grid_sample(
        pers_bchw, grid_batch, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # Convert BCHW → BHWC
    patch_360 = out_bchw.permute(0, 2, 3, 1)

    # Scale back to [0,1] range for ComfyUI output format
    patch_360 = patch_360 / 255.0

    # -------------------------
    # Feathering (optional)
    # -------------------------
    if feather > 0:
        k = int(feather) * 2 + 1

        # Prepare mask for erosion [1, 1, H, W] - convert to [0,1] range
        m = (mask_t.to(torch.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        # Erosion via negative max pooling
        m_eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=k // 2)

        # Gaussian blur cache
        gauss_key = (k, device_str)
        if gauss_key not in p2e_and_blend_torch._gauss_cache:
            # Generate 1D Gaussian kernel
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
            xs = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
            g = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
            g = (g / g.sum()).to(torch.float32)
            kx = g.view(1, 1, 1, k)
            ky = g.view(1, 1, k, 1)
            p2e_and_blend_torch._gauss_cache[gauss_key] = (kx, ky)
        else:
            kx, ky = p2e_and_blend_torch._gauss_cache[gauss_key]

        # Apply separable Gaussian blur
        m_pad = F.pad(m_eroded, (k // 2, k // 2, 0, 0), mode="reflect")
        m_blur = F.conv2d(m_pad, kx)
        m_pad = F.pad(m_blur, (0, 0, k // 2, k // 2), mode="reflect")
        m_blur = F.conv2d(m_pad, ky)

        alpha = m_blur.squeeze(0).squeeze(0)  # [H, W]
    else:
        alpha = mask_t.to(torch.float32) / 255.0  # [H, W] in [0,1]

    # -------------------------
    # Blending
    # -------------------------
    # Expand alpha for batch and channels [B, H, W, C]
    num_channels = equi_base_t.shape[3]
    alpha_bhwc = alpha.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
    alpha_bhwc = alpha_bhwc.expand(batch_size, -1, -1, num_channels)

    # Alpha blend: merged = base * (1 - alpha) + patch * alpha
    merged = equi_base_t * (1.0 - alpha_bhwc) + patch_360 * alpha_bhwc

    # Prepare mask output [B, H, W]
    mask_out = alpha.unsqueeze(0).expand(batch_size, -1, -1)

    # Restore to original container/dtype/layout
    merged_out = _restore_from_bhwc(merged, equi_meta)
    patch_out = _restore_from_bhwc(patch_360, persp_meta)
    mask_out = _restore_from_bhwc(mask_out.unsqueeze(-1), equi_meta)

    return merged_out, patch_out, mask_out


# -------------------------
# ComfyUI Node Class
# -------------------------


class P2EAndBlendNode:
    """
    Perspective to Equirectangular with Blending

    Projects a perspective image onto an equirectangular base image with optional
    feathering for seamless blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "perspective": (
                    "IMAGE",
                    {
                        "tooltip": "The perspective image to project onto the equirectangular base"
                    },
                ),
                "equi_base": (
                    "IMAGE",
                    {
                        "tooltip": "The equirectangular panorama base image onto which the perspective will be projected"
                    },
                ),
                "fov_w": (
                    "FLOAT",
                    {
                        "default": 140.0,
                        "min": 1.0,
                        "max": 180.0,
                        "step": 0.1,
                        "tooltip": "Horizontal field of view in degrees",
                    },
                ),
                "fov_h": (
                    "FLOAT",
                    {
                        "default": 140.0,
                        "min": 1.0,
                        "max": 180.0,
                        "step": 0.1,
                        "tooltip": "Vertical field of view in degrees",
                    },
                ),
                "u_deg": (
                    "FLOAT",
                    {
                        "default": 180,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 0.1,
                        "tooltip": "Horizontal rotation in degrees (yaw). Range: -180 to 180",
                    },
                ),
                "v_deg": (
                    "FLOAT",
                    {
                        "default": -70,
                        "min": -90.0,
                        "max": 90.0,
                        "step": 0.1,
                        "tooltip": "Vertical rotation in degrees (pitch). Range: -90 to 90",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Feather/blur radius for blending edge",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("merged", "patch_360", "mask")
    OUTPUT_TOOLTIPS = (
        "The final blended result combining the equirectangular base with the projected perspective image",
        "The perspective image projected onto equirectangular coordinates (before blending)",
        "The blending mask showing the projection area with feathering applied",
    )
    FUNCTION = "process"
    CATEGORY = "p2e"

    def process(self, perspective, equi_base, fov_w, fov_h, u_deg, v_deg, feather):
        """
        Process perspective-to-equirectangular projection and blending.

        Args:
            perspective: torch.Tensor or np.ndarray, image in HWC/CHW/BHWC/BCHW
            equi_base: torch.Tensor or np.ndarray, image in HWC/CHW/BHWC/BCHW
            fov_w: float, horizontal FOV in degrees
            fov_h: float, vertical FOV in degrees
            u_deg: float, horizontal rotation in degrees
            v_deg: float, vertical rotation in degrees
            feather: int, feather radius in pixels

        Returns:
            tuple: (merged, patch_360, mask)
        """
        # Call pure torch function
        merged, patch_360, mask = p2e_and_blend_torch(
            perspective=perspective,
            equi_base=equi_base,
            fov_deg=(fov_w, fov_h),
            u_deg=u_deg,
            v_deg=v_deg,
            feather=feather,
            device=None,  # Let function determine device
        )

        return (merged, patch_360, mask)


# -------------------------
# Node Registration
# -------------------------

NODE_CLASS_MAPPINGS = {
    "Perspective to Equirectangular": P2EAndBlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Perspective to Equirectangular": "Perspective to Equirectangular",
}
