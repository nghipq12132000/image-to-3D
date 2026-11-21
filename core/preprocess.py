"""
2D Image ROI Preprocessing & Automatic Silhouette Mask Extraction for Image-to-3D Studio.
"""
import io
import base64
import cv2
import numpy as np
import torch


def extract_foreground_mask_and_bbox(img_rgb, mask_mode="auto"):
    """
    Automatically extracts the binary silhouette mask and tight bounding box [x1, y1, x2, y2]
    of the furniture object from an input RGB/RGBA image.
    """
    h, w = img_rgb.shape[:2]

    if mask_mode == "full":
        mask = np.ones((h, w), dtype=np.uint8) * 255
        bbox = [0, 0, w, h]
        return mask, bbox

    # Estimate background color from the 4 border strips
    border_px = max(2, min(h, w) // 25)
    top_strip = img_rgb[:border_px, :, :].reshape(-1, 3)
    bot_strip = img_rgb[-border_px:, :, :].reshape(-1, 3)
    left_strip = img_rgb[:, :border_px, :].reshape(-1, 3)
    right_strip = img_rgb[:, -border_px:, :].reshape(-1, 3)
    border_samples = np.concatenate([top_strip, bot_strip, left_strip, right_strip], axis=0).astype(np.float32)
    bg_median = np.median(border_samples, axis=0)

    # Color distance from border background + edge saliency
    diff = np.linalg.norm(img_rgb.astype(np.float32) - bg_median[None, None, :], axis=-1)
    diff_norm = np.clip((diff / (np.max(diff) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)

    _, otsu_mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # GrabCut refinement when image has rich interior texture
    try:
        gc_mask = np.zeros((h, w), np.uint8)
        pad_x, pad_y = max(4, int(w * 0.06)), max(4, int(h * 0.06))
        rect = (pad_x, pad_y, max(10, w - 2 * pad_x), max(10, h - 2 * pad_y))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(img_bgr, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        fg_gc = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg_ratio = np.mean(fg_gc > 0)
        if 0.05 < fg_ratio < 0.92:
            combined = cv2.bitwise_or(fg_gc, otsu_mask)
        else:
            combined = otsu_mask
    except Exception:
        combined = otsu_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep central/largest connected component if valid
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest_idx, cv2.CC_STAT_AREA] > 0.03 * (h * w):
            cleaned = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    ys, xs = np.where(cleaned > 127)
    if len(xs) < int(0.02 * h * w):
        cleaned = np.ones((h, w), dtype=np.uint8) * 255
        bbox = [0, 0, w, h]
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return cleaned, bbox


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
    res_msk = cv2.resize(sq_msk, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    res_msk_bin = (res_msk > 127).astype(np.uint8)
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


def preprocess_image_bytes(image_bytes, device, mask_mode="auto", target_size=224):
    """
    Full 2D preprocessing pipeline from raw image bytes -> PyTorch tensors & preview data URIs.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError("Không thể giải mã tệp hình ảnh. Vui lòng chọn ảnh PNG, JPG hoặc WEBP hợp lệ.")

    # Check if RGBA with alpha channel transparency
    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        alpha = img_raw[:, :, 3]
        img_rgb = cv2.cvtColor(img_raw[:, :, :3], cv2.COLOR_BGR2RGB)
        if np.min(alpha) < 245 and np.max(alpha) > 10 and mask_mode == "auto":
            mask_gray = np.where(alpha > 127, 255, 0).astype(np.uint8)
            ys, xs = np.where(mask_gray > 0)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) > 10 else [0, 0, img_rgb.shape[1], img_rgb.shape[0]]
        else:
            mask_gray, bbox = extract_foreground_mask_and_bbox(img_rgb, mask_mode=mask_mode)
    elif img_raw.ndim == 2:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        mask_gray, bbox = extract_foreground_mask_and_bbox(img_rgb, mask_mode=mask_mode)
    else:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        mask_gray, bbox = extract_foreground_mask_and_bbox(img_rgb, mask_mode=mask_mode)

    roi_rgb, roi_msk = crop_and_pad_roi(img_rgb, mask_gray, bbox, target_size=target_size)

    img_f = roi_rgb.astype(np.float32) / 255.0
    msk_f = roi_msk.astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_f - mean) / std

    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    msk_t = torch.from_numpy(msk_f).unsqueeze(0).unsqueeze(0).to(device)        # (1, 1, 224, 224)

    return {
        "img_tensor": img_t,
        "mask_tensor": msk_t,
        "roi_rgb_b64": encode_image_base64(roi_rgb, is_mask=False),
        "mask_b64": encode_image_base64(roi_msk, is_mask=True),
        "orig_size": [int(img_rgb.shape[1]), int(img_rgb.shape[0])],
        "foreground_coverage_pct": round(float(np.mean(roi_msk > 0) * 100.0), 1),
    }
