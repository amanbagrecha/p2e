import numpy as np
from PIL import Image

equi_img_np  = np.asarray(Image.open(r"C:\Users\Admin\Downloads\full-workflow\1001223444465_1221212321131_ladybug_panoramic_000271.jpg"))
perspective_np = np.asarray(Image.open(r"C:\Users\Admin\Downloads\full-workflow\1001223444465_1221212321131_ladybug_panoramic_000271_perspective_comfyui.png"))


def p2e_and_blend(perspective, equi_base, fov_deg, u_deg, v_deg, feather=0, device=None):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hasattr(p2e_and_blend, "_grid_cache"):
        p2e_and_blend._grid_cache = {}
    if not hasattr(p2e_and_blend, "_gauss_cache"):
        p2e_and_blend._gauss_cache = {}

    h_eq, w_eq = equi_base.shape[:2]
    h_p, w_p = perspective.shape[:2]
    fov_w, fov_h = float(fov_deg[0]), float(fov_deg[1])

    cache_key = (h_eq, w_eq, h_p, w_p, fov_w, fov_h, float(u_deg), float(v_deg), device)

    if cache_key not in p2e_and_blend._grid_cache:
        eq_y, eq_x = torch.meshgrid(
            torch.arange(h_eq, device=device, dtype=torch.float32),
            torch.arange(w_eq, device=device, dtype=torch.float32),
            indexing="ij"
        )

        theta = (eq_x / w_eq - 0.5) * (2.0 * torch.pi)
        phi = -(eq_y / h_eq - 0.5) * torch.pi

        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi)
        z = torch.cos(phi) * torch.cos(theta)
        xyz = torch.stack([x, y, z], dim=-1)

        u = torch.deg2rad(torch.tensor(float(u_deg), device=device, dtype=torch.float32))
        v = torch.deg2rad(torch.tensor(float(-v_deg), device=device, dtype=torch.float32))

        cu, su = torch.cos(u), torch.sin(u)
        cv, sv = torch.cos(v), torch.sin(v)

        z0 = torch.tensor(0.0, device=device, dtype=torch.float32)
        o1 = torch.tensor(1.0, device=device, dtype=torch.float32)

        Ry = torch.stack([
            torch.stack([ cu, z0, -su]),
            torch.stack([ z0, o1,  z0]),
            torch.stack([ su, z0,  cu]),
        ], dim=0)

        Rx = torch.stack([
            torch.stack([ o1, z0,  z0]),
            torch.stack([ z0,  cv, -sv]),
            torch.stack([ z0,  sv,  cv]),
        ], dim=0)

        R = Ry @ Rx
        xyz_cam = xyz @ R

        valid = xyz_cam[..., 2] > 0
        z_safe = torch.where(valid, xyz_cam[..., 2], torch.ones_like(xyz_cam[..., 2]))

        x_p = xyz_cam[..., 0] / z_safe
        y_p = xyz_cam[..., 1] / z_safe

        tan_h = torch.tan(torch.deg2rad(torch.tensor(fov_w, device=device, dtype=torch.float32)) / 2.0)
        tan_v = torch.tan(torch.deg2rad(torch.tensor(fov_h, device=device, dtype=torch.float32)) / 2.0)

        in_fov = valid & (x_p.abs() <= tan_h) & (y_p.abs() <= tan_v)

        u_px = (x_p / tan_h + 1.0) * 0.5 * w_p
        v_px = (-y_p / tan_v + 1.0) * 0.5 * h_p

        map_x = torch.where(in_fov, u_px, torch.full_like(u_px, -1.0))
        map_y = torch.where(in_fov, v_px, torch.full_like(v_px, -1.0))

        grid_x = (map_x / (w_p - 1.0)) * 2.0 - 1.0
        grid_y = (map_y / (h_p - 1.0)) * 2.0 - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        mask_t = (in_fov.to(torch.uint8) * 255)

        p2e_and_blend._grid_cache[cache_key] = (grid, mask_t)
    else:
        grid, mask_t = p2e_and_blend._grid_cache[cache_key]

    dtype_in = perspective.dtype
    is_color = (perspective.ndim == 3)

    pers_t = torch.from_numpy(perspective).to(device).float()
    if is_color:
        pers_t = pers_t.permute(2, 0, 1).unsqueeze(0)
    else:
        pers_t = pers_t.unsqueeze(0).unsqueeze(0)

    out = F.grid_sample(
        pers_t,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    if is_color:
        patch_360_t = out[0].permute(1, 2, 0)
    else:
        patch_360_t = out[0, 0]

    patch_360 = patch_360_t.detach().cpu().numpy()
    if dtype_in == np.uint8:
        patch_360 = np.clip(patch_360, 0, 255).astype(np.uint8)
    elif dtype_in == np.uint16:
        patch_360 = np.clip(patch_360, 0, 65535).astype(np.uint16)
    else:
        patch_360 = patch_360.astype(dtype_in)

    mask_np = mask_t.detach().cpu().numpy()

    # -------------------------
    # Step C: Torch-only blend
    # -------------------------
    if feather > 0:
        k = int(feather) * 2 + 1

        m = (mask_t.to(torch.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # 1x1xH xW

        m_eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=k // 2)

        gauss_key = (k, device)
        if gauss_key not in p2e_and_blend._gauss_cache:
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # OpenCV sigma when sigma=0
            xs = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
            g = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
            g = (g / g.sum()).to(torch.float32)
            kx = g.view(1, 1, 1, k)
            ky = g.view(1, 1, k, 1)
            p2e_and_blend._gauss_cache[gauss_key] = (kx, ky)
        else:
            kx, ky = p2e_and_blend._gauss_cache[gauss_key]

        m_pad = F.pad(m_eroded, (k // 2, k // 2, 0, 0), mode="reflect")
        m_blur = F.conv2d(m_pad, kx)

        m_pad = F.pad(m_blur, (0, 0, k // 2, k // 2), mode="reflect")
        m_blur = F.conv2d(m_pad, ky)

        alpha_t = m_blur.squeeze(0).squeeze(0)  # HxW in [0,1]
    else:
        alpha_t = (mask_t.to(torch.float32) / 255.0)  # HxW

    base_t = torch.from_numpy(equi_base).to(device).float()
    patch_t = patch_360_t  # already torch on device

    if base_t.ndim == 3:
        alpha_t3 = alpha_t.unsqueeze(-1)
    else:
        alpha_t3 = alpha_t

    result_t = base_t * (1.0 - alpha_t3) + patch_t * alpha_t3

    if equi_base.dtype == np.uint8:
        result = result_t.clamp(0, 255).to(torch.uint8).cpu().numpy()
    elif equi_base.dtype == np.uint16:
        result = result_t.clamp(0, 65535).to(torch.uint16).cpu().numpy()
    else:
        result = result_t.to(torch.from_numpy(equi_base).dtype).cpu().numpy()

    return result, patch_360, mask_np




# Example:
merged, patch_360, mask = p2e_and_blend(
    perspective=perspective_np,
    equi_base=equi_img_np,
    fov_deg=(140, 140),
    u_deg=180,
    v_deg=-70,
    feather=10,
)


