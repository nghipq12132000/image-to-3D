"""
2D Image ROI Preprocessing & Shadow-Invariant Multi-Regime Silhouette Mask Extraction for Image-to-3D Studio.
"""
import base64
import cv2
import numpy as np
import torch


def extract_foreground_mask_and_bbox(img_rgb, mask_mode="auto", sensitivity=1.0):
    """
    Multi-Regime Smart Furniture Silhouette Extractor:
      1. Regime 'studio' (Uniform / Catalog Studio Background):
         Automatically triggered when border variance is low (or mask_mode == 'studio').
         Captures subtle furniture boundaries (even light/white furniture on white studio backgrounds)
         while preserving see-through holes.
      2. Regime 'indoor' (Real-World Indoor Scene with Wall, Baseboard & Wooden Floor):
         Uses Shadow-Invariant Row-Chromaticity & Photometric Ratio modeling + Vertical Leg Ridge detection:
           - Cancels horizontal wall-floor baseboards and ambient cast shadows on walls/floors
             (where 0.61 <= L/L_bg <= 1.04 and chromaticity (a*, b*) matches the row background).
           - Preserves hollow gaps between chair backrest slats and under-seat stretcher rails.
           - Traces vertical wooden chair legs all the way down across similar-colored wooden floors
             via local horizontal ridge contrast.
    """
    h, w = img_rgb.shape[:2]
    if mask_mode == "full":
        return np.ones((h, w), dtype=np.uint8) * 255, [0, 0, w, h]

    sens = float(np.clip(sensitivity, 0.4, 2.0))
    thresh_scale = 1.0 / sens

    # Ignore artificial pure-black square-padding bars (RGB <= 4) if present
    non_black = np.any(img_rgb > 4, axis=-1)
    ys_nb, xs_nb = np.where(non_black)
    if len(xs_nb) > int(0.1 * h * w):
        vx1, vx2 = int(xs_nb.min()), int(xs_nb.max()) + 1
        vy1, vy2 = int(ys_nb.min()), int(ys_nb.max()) + 1
    else:
        vx1, vx2, vy1, vy2 = 0, w, 0, h

    sub_rgb = img_rgb[vy1:vy2, vx1:vx2]
    sh, sw = sub_rgb.shape[:2]

    lab = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    bw = max(3, int(sw * 0.08))

    lr_L = np.concatenate([L[:, :bw], L[:, sw - bw:]], axis=1)
    lr_A = np.concatenate([A[:, :bw], A[:, sw - bw:]], axis=1)
    lr_B = np.concatenate([B[:, :bw], B[:, sw - bw:]], axis=1)

    border_L_std = float(np.std(lr_L))
    border_AB_std = float(np.std(lr_A) + np.std(lr_B))
    is_studio_bg = (mask_mode == "studio") or (mask_mode == "auto" and border_L_std < 6.5 and border_AB_std < 4.5)

    if is_studio_bg:
        bg_L = float(np.median(lr_L))
        bg_A = float(np.median(lr_A))
        bg_B = float(np.median(lr_B))
        dE = np.sqrt((L - bg_L) ** 2 + 2.5 * (A - bg_A) ** 2 + 2.5 * (B - bg_B) ** 2)
        gx = np.abs(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
        grad = np.sqrt(gx ** 2 + gy ** 2)

        sub_mask = ((dE > (3.8 * thresh_scale)) | (grad > (10.0 * thresh_scale))).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_CLOSE, k_close, iterations=2)

        padded = cv2.copyMakeBorder(sub_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        ff_mask = np.zeros((sh + 4, sw + 4), dtype=np.uint8)
        inv = cv2.bitwise_not(padded)
        cv2.floodFill(inv, ff_mask, (0, 0), 0)
        interior = inv[1:-1, 1:-1] > 0
        is_true_bg_hole = (dE < (2.5 * thresh_scale)) & (grad < 5.0)
        sub_mask = np.where((sub_mask > 0) | (interior & (~is_true_bg_hole)), 255, 0).astype(np.uint8)

    else:
        # Shadow-Invariant Row-Chromaticity & Vertical Leg Ridge Extractor
        L_bg = cv2.medianBlur(np.median(lr_L, axis=1, keepdims=True).astype(np.uint8), 5).astype(np.float32)
        A_bg = cv2.medianBlur(np.median(lr_A, axis=1, keepdims=True).astype(np.uint8), 5).astype(np.float32)
        B_bg = cv2.medianBlur(np.median(lr_B, axis=1, keepdims=True).astype(np.uint8), 5).astype(np.float32)

        chroma_dist = np.sqrt((A - A_bg) ** 2 + (B - B_bg) ** 2)
        bg_chroma_mag = np.sqrt((A_bg - 128.0) ** 2 + (B_bg - 128.0) ** 2)
        is_colored_floor_row = bg_chroma_mag > 8.0

        lum_ratio = L / np.maximum(L_bg, 10.0)
        lum_diff  = L - L_bg

        gx = np.abs(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
        lr_gy = np.median(np.concatenate([gy[:, :bw], gy[:, sw - bw:]], axis=1), axis=1, keepdims=True)
        is_baseboard_row = (lr_gy > 8.0) & (gy > 1.2 * gx)

        sigma_x = max(4.0, sw * 0.06)
        L_horiz_bg = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma_x, sigmaY=0.8)
        local_ridge_diff = L - L_horiz_bg

        is_chroma_fg = chroma_dist > (8.2 * thresh_scale)
        is_bright_fg = (lum_diff > (11.5 * thresh_scale)) & (chroma_dist > (4.5 * thresh_scale))
        is_floor_leg_ridge = is_colored_floor_row & (~is_baseboard_row) & (local_ridge_diff > (13.5 * thresh_scale)) & (lum_diff > 4.5)
        is_dark_albedo_fg = lum_ratio < (0.60 / np.sqrt(thresh_scale))

        fg_mask = (is_chroma_fg | is_bright_fg | is_floor_leg_ridge | is_dark_albedo_fg)

        # Reject cast shadows on walls & floors
        is_cast_shadow_bg = (lum_ratio >= 0.61) & (lum_ratio <= 1.04) & (chroma_dist < (7.8 * thresh_scale)) & (~is_floor_leg_ridge)
        fg_mask = fg_mask & (~is_cast_shadow_bg)

        # Reject horizontal wall-floor baseboard line
        fg_mask = fg_mask & (~(is_baseboard_row & (chroma_dist < 14.5) & (lum_ratio > 0.60)))

        px_m = max(2, int(sw * 0.025))
        py_m = max(2, int(sh * 0.025))
        fg_mask[:, :px_m] = False
        fg_mask[:, sw - px_m:] = False
        fg_mask[:py_m, :] = False

        sub_mask = (fg_mask.astype(np.uint8)) * 255

        horiz_open = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        vert_open  = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        sub_mask = cv2.bitwise_or(horiz_open, vert_open)

        k_vert_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_CLOSE, k_vert_close, iterations=1)

    # Keep main furniture connected component and vertically aligned legs/parts
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sub_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_idx = 1 + int(np.argmax(areas))
        mx1 = stats[main_idx, cv2.CC_STAT_LEFT]
        mx2 = mx1 + stats[main_idx, cv2.CC_STAT_WIDTH]
        my1 = stats[main_idx, cv2.CC_STAT_TOP]

        clean = np.zeros_like(sub_mask)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            cx, cy = centroids[lbl]
            if lbl == main_idx or (area >= max(4, int(0.0012 * sh * sw)) and (mx1 - 0.03 * sw) <= cx <= (mx2 + 0.03 * sw) and cy >= my1):
                clean[labels == lbl] = 255
        sub_mask = clean

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[vy1:vy2, vx1:vx2] = sub_mask

    ys, xs = np.where(full_mask > 127)
    if len(xs) < int(0.015 * h * w):
        full_mask[vy1:vy2, vx1:vx2] = 255
        bbox = [vx1, vy1, vx2, vy2]
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return full_mask, bbox


def crop_and_pad_roi(img_rgb, mask_gray, bbox, target_size=224, margin_ratio=0.10):
    """
    Crops the object ROI with a 10% margin, applies aspect-ratio preserving square padding,
    and resizes to 224x224 (exact Pix3D training pipeline protocol).
    """
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    side = int(max(bw, bh) * (1.0 + 2.0 * margin_ratio))

    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(w, int(cx + side / 2))
    ny2 = min(h, int(cy + side / 2))

    crop_img = img_rgb[ny1:ny2, nx1:nx2]
    crop_msk = mask_gray[ny1:ny2, nx1:nx2]

    ch, cw = crop_img.shape[:2]
    max_c = max(ch, cw, 1)
    pad_top = (max_c - ch) // 2
    pad_bot = max_c - ch - pad_top
    pad_left = (max_c - cw) // 2
    pad_right = max_c - cw - pad_left

    sq_img = cv2.copyMakeBorder(crop_img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    sq_msk = cv2.copyMakeBorder(crop_msk, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    res_img = cv2.resize(sq_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    res_msk = cv2.resize(sq_msk, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    res_msk_bin = (res_msk > 115).astype(np.uint8)
    return res_img, res_msk_bin


def encode_image_base64(img_rgb_or_gray, is_mask=False):
    """Encodes a numpy RGB or grayscale array into a data:image/png;base64 URI."""
    if is_mask:
        vis = (img_rgb_or_gray * 255).astype(np.uint8) if img_rgb_or_gray.max() <= 1 else img_rgb_or_gray.astype(np.uint8)
        ok, buf = cv2.imencode(".png", vis)
    else:
        bgr = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def preprocess_image_bytes(image_bytes, device, mask_mode="auto", sensitivity=1.0, target_size=224):
    """
    Full 2D preprocessing pipeline from raw image bytes -> PyTorch tensors & preview data URIs.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError("Không thể giải mã tệp hình ảnh. Vui lòng chọn ảnh PNG, JPG hoặc WEBP hợp lệ.")

    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        alpha = img_raw[:, :, 3]
        img_rgb = cv2.cvtColor(img_raw[:, :, :3], cv2.COLOR_BGR2RGB)
        if np.min(alpha) < 245 and np.max(alpha) > 10 and mask_mode in ("auto", "indoor"):
            mask_gray = np.where(alpha > 127, 255, 0).astype(np.uint8)
            ys, xs = np.where(mask_gray > 0)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 10 else [0, 0, img_rgb.shape[1], img_rgb.shape[0]]
        else:
            mask_gray, bbox = extract_foreground_mask_and_bbox(img_rgb, mask_mode=mask_mode, sensitivity=sensitivity)
    elif img_raw.ndim == 2:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        mask_gray, bbox = extract_foreground_mask_and_bbox(img_rgb, mask_mode=mask_mode, sensitivity=sensitivity)
    else:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        mask_gray, bbox = extract_foreground_mask_and_bbox(img_rgb, mask_mode=mask_mode, sensitivity=sensitivity)

    roi_rgb, roi_msk = crop_and_pad_roi(img_rgb, mask_gray, bbox, target_size=target_size)

    img_f = roi_rgb.astype(np.float32) / 255.0
    msk_f = roi_msk.astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_f - mean) / std

    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    msk_t = torch.from_numpy(msk_f).unsqueeze(0).unsqueeze(0).to(device)

    return {
        "img_tensor": img_t,
        "mask_tensor": msk_t,
        "roi_rgb_b64": encode_image_base64(roi_rgb, is_mask=False),
        "mask_b64": encode_image_base64(roi_msk, is_mask=True),
        "orig_size": [int(img_rgb.shape[1]), int(img_rgb.shape[0])],
        "foreground_coverage_pct": round(float(np.mean(roi_msk > 0) * 100.0), 1),
    }
