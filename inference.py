"""Inpainting — Full Stack (LoRA + ControlNet + IP-Adapter)."""

from __future__ import annotations
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

CANNY_LOW      = 80
CANNY_HIGH     = 150
MASK_DILATE_PX = 5
PROMPT     = "a car, realistic, high quality, detailed"
NEG_PROMPT = "blurry, distorted, artifacts, deformed"

# ── Letterbox helpers ──────────────────────────────────────────────────────────

def _letterbox_pad(img: Image.Image, size: int = 512) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h    = img.size
    scale   = size / max(w, h)
    new_w   = int(round(w * scale))
    new_h   = int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2

    canvas = Image.new(img.mode, (size, size), 0)
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, (pad_top, pad_left, new_h, new_w)

def _unpad_resize(img_padded: Image.Image,
                  pad_info: tuple[int, int, int, int],
                  orig_size: tuple[int, int]) -> Image.Image:
    pad_top, pad_left, new_h, new_w = pad_info
    cropped = img_padded.crop((pad_left, pad_top, pad_left + new_w, pad_top + new_h))
    return cropped.resize(orig_size, Image.Resampling.LANCZOS)

# ── Canny & IP helpers ────────────────────────────────────────────────────────

def extract_canny_masked(
    image_pil: Image.Image, mask_pil: Image.Image
) -> tuple[Image.Image, dict]:
    img   = np.array(image_pil.convert("RGB"))
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    msk   = np.array(mask_pil.convert("L"))
    if MASK_DILATE_PX > 0:
        k   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MASK_DILATE_PX * 2 + 1, MASK_DILATE_PX * 2 + 1),
        )
        msk = cv2.dilate(msk, k, iterations=1)
    edges[msk > 127] = 0

    outside_mask = msk <= 127
    total_px     = int(outside_mask.sum())
    edge_px      = int((edges[outside_mask] > 0).sum())
    edge_density = edge_px / total_px * 100 if total_px > 0 else 0.0

    canny_info = {
        "threshold_low":  CANNY_LOW,
        "threshold_high": CANNY_HIGH,
        "edge_density_pct": round(edge_density, 2),
    }

    canny_img = Image.fromarray(np.stack([edges] * 3, axis=2))
    return canny_img, canny_info

def extract_visible_patch(image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    img     = np.array(image_pil.convert("RGB"))
    msk     = np.array(mask_pil.convert("L"))
    visible = msk < 128
    mean_c  = (
        img[visible].mean(axis=0).astype(np.uint8)
        if visible.sum() > 0
        else np.array([128, 128, 128], dtype=np.uint8)
    )
    patch           = img.copy()
    patch[~visible] = mean_c
    return Image.fromarray(patch)

# ── Main inpaint ──────────────────────────────────────────────────────────────

def inpaint(
    pipe,
    img_pil: Image.Image,
    mask_pil: Image.Image,
    steps: int      = 20,
    guidance: float = 7.5,
    cn_scale: float = 0.7,
    ip_scale: float = 0.5,
    seed: int       = 42,
):
    orig_w, orig_h = img_pil.size
    pipe.set_ip_adapter_scale(ip_scale)

    # 1. Letterbox anh goc + mask ve 512x512
    img_512, pad_info = _letterbox_pad(img_pil, size=512)
    mask_512_rgb, _   = _letterbox_pad(mask_pil.convert("RGB"), size=512)
    mask_512          = mask_512_rgb.convert("L")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # 2. Inference
    canny_img, canny_info = extract_canny_masked(img_512, mask_512)
    try:
        with torch.inference_mode():
            result_512 = pipe(
                prompt=PROMPT,
                negative_prompt=NEG_PROMPT,
                image=img_512,
                mask_image=mask_512,
                control_image=canny_img,
                ip_adapter_image=extract_visible_patch(img_512, mask_512),
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=cn_scale,
                generator=generator,
            ).images[0]
    except torch.cuda.OutOfMemoryError:
        return img_pil, "❌ CUDA OOM — Giảm Steps lại", None, None
    except Exception as exc:
        return img_pil, f"❌ Lỗi: {exc}", None, None

    # 3. Unpad -> resize ve kich thuoc goc
    result_orig = _unpad_resize(result_512, pad_info, (orig_w, orig_h))

    # 4. Composite: chi paste vung mask vao anh goc (giữ nguyên viền)
    mask_orig_blur = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))
    final = img_pil.copy()
    final.paste(result_orig, mask=mask_orig_blur)

    return final, "✅ Khôi phục thành công!", canny_info, canny_img