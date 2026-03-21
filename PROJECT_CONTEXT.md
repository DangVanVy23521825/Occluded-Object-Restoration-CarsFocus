# PROJECT CONTEXT — CS331 Occluded Object Reconstruction
> Paste file này vào đầu mỗi cuộc hội thoại với Cursor agent.
> Đọc kỹ toàn bộ trước khi viết bất kỳ dòng code nào.

---

## 1. Bài toán một câu
Cho ảnh xe hơi bị che khuất một phần, tái tạo vùng bị thiếu sao cho kết quả nhất quán về hình dạng, màu sắc và chất liệu bề mặt với phần còn lại của xe.

---

## 2. Kiến trúc pipeline (3 module, tích hợp dần theo tuần)

```
x_occ ──────────────────────────────────────────► x_hat
         │              │              │
    [Module A]     [Module B]     [Module C]
  SD Inpainting  ControlNet-     IP-Adapter
   (backbone)    Canny (shape)  (surface/color)
         │              │              │
       Tuần 2         Tuần 3         Tuần 4
      BASELINE       +shape        +surface
```

**Ký hiệu toán học — dùng nhất quán trong toàn bộ code và comment:**

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `x_gt` | ảnh xe gốc hoàn chỉnh (ground truth) |
| `x_occ` | ảnh bị che = `x_gt ⊙ (1 - M)` |
| `M` | binary mask — 1 = bị che, 0 = visible |
| `C` | Canny edge map — **trích từ `x_occ`, KHÔNG từ `x_gt`** |
| `P` | visible patch = `x_gt ⊙ (1-M)` → reference cho IP-Adapter |
| `x_hat` | ảnh phục dựng (output) |

> ⚠️ **Rule cứng số 1:** `C` (Canny) PHẢI lấy từ `x_occ`. Dùng `x_gt` là oracle bias — sai về mặt khoa học.

---

## 3. Checkpoints & thư viện

| Module | Checkpoint (HuggingFace) | Thư viện |
|--------|--------------------------|----------|
| SD Inpainting | `stabilityai/stable-diffusion-2-inpainting` | `diffusers >= 0.24` |
| ControlNet | `lllyasviel/control_v11p_sd21_canny` | `controlnet-aux` |
| IP-Adapter | `h94/IP-Adapter` → `ip-adapter-plus_sd15.bin` | `diffusers >= 0.24` |

---

## 4. Dataset & Data pipeline

- **Stanford Cars** — 16,185 ảnh, 196 dòng xe (`torchvision.datasets.StanfordCars`)
- **Synthetic occlusion** — đặt COCO objects lên xe, IoU target **0.15–0.40**
- Mỗi sample gồm 3 file: `{id}_gt.jpg` | `{id}_occ.jpg` | `{id}_mask.png`
- Resize về **512×512** bằng Letterbox (giữ aspect ratio, padding trung tính)
- Tập test chuẩn: **200 ảnh**, stratified theo car class

---

## 5. Evaluation

Metrics tính **chỉ trên vùng mask** (không phải toàn ảnh):

| Metric | Thư viện | Hướng tốt |
|--------|----------|-----------|
| LPIPS | `lpips`, net=`alex` | ↓ thấp hơn |
| SSIM | `scikit-image` | ↑ cao hơn |
| FID | `clean-fid` | ↓ thấp hơn |

**Ablation study — 4 cấu hình cần so sánh:**
```
Config A: SD only                              ← Baseline
Config B: SD + ControlNet-Canny               ← +Shape
Config C: SD + ControlNet + IP-Adapter        ← +Surface
Config D: SD + ControlNet + Text Color Prompt ← Fallback nếu C OOM
```

Kết quả phân tầng theo occlusion ratio: `light < 0.20` | `medium 0.20–0.40` | `heavy > 0.40`

---

## 6. Nền tảng & môi trường

- **Kaggle Notebook** — GPU T4 x2 hoặc P100 (16GB VRAM)
- **Python thuần** — không dùng Lightning, không dùng Trainer
- Tất cả output ghi vào `/kaggle/working/`
- Path detection mẫu:
```python
from pathlib import Path
ROOT = Path("/kaggle/working")
DATA_DIR  = ROOT / "data"
CKPT_DIR  = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
```

---

## 7. VRAM rules (BẮT BUỘC với mọi file liên quan model)

```python
# Luôn làm đủ 4 bước này sau khi build pipeline
pipe = pipe.to("cuda", dtype=torch.float16)
pipe.enable_xformers_memory_efficient_attention()  # ưu tiên
pipe.enable_attention_slicing()                    # fallback
pipe.enable_vae_slicing()                          # thêm ~1GB

# Luôn wrap inference
with torch.no_grad(), torch.autocast("cuda"):
    result = pipe(..., generator=generator).images[0]
torch.cuda.empty_cache()
```

---

## 8. Reproducibility (BẮT BUỘC mọi notebook và script)

```python
SEED = 42
import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
generator = torch.Generator(device="cuda").manual_seed(SEED)
# Truyền generator= vào MỌI pipeline call
```

---

## 9. Cấu trúc thư mục

```
/kaggle/working/
├── configs/config.yaml         ← MỌI hyperparameter ở đây, không hardcode
├── data/
│   ├── synthetic_occlusion.py  ← tạo (x_gt, x_occ, M)
│   ├── dataset.py
│   └── preprocess.py
├── models/
│   ├── pipeline_builder.py     ← factory: build_pipeline(use_controlnet, use_ip_adapter)
│   ├── inpainting.py           ← Module A
│   ← Module B
│   └── ip_adapter.py           ← Module C + fallback text color
├── evaluation/
│   ├── metrics.py              ← compute_masked_metrics(x_gt, x_hat, mask)
│   └── run_eval.py             ← chạy ablation, lưu CSV
├── app/
│   ├── app.py                  ← Gradio 2-tab demo
│   └── pipeline_cache.py       ← singleton, không reload model
└── results/
    ├── metrics/                ← CSV mỗi config
    └── ablation_table.csv
```

---

## 10. Những điều KHÔNG được làm

| Sai | Đúng |
|-----|------|
| Trích Canny từ `x_gt` | Trích Canny từ `x_occ` |
| Hardcode path hay hyperparameter | Đọc từ `config.yaml` |
| Bỏ `torch.no_grad()` khi inference | Luôn wrap `no_grad + autocast` |
| Để notebook crash mất hết kết quả | Save checkpoint mỗi 50 ảnh |
| Reload pipeline mỗi Gradio request | Cache pipeline bằng `gr.State()` |
| Tính metrics trên toàn ảnh | Tính chỉ trên vùng `mask == 1` |
| Raise exception khi IP-Adapter OOM | Fallback sang text color prompt |

---

## 11. IP-Adapter fallback (quan trọng)

Nếu VRAM < 12GB khi chạy cả 3 module, tự động dùng:
```python
def get_color_prompt(visible_patch: np.ndarray) -> str:
    """Trích màu dominant từ visible patch → text prompt thay IP-Adapter."""
    # VD output: "a car with dark red metallic paint, glossy surface"
    ...
```
Đây là **Config D** trong ablation, không phải lỗi — ghi rõ trong báo cáo.

---

## 12. Demo (Gradio)

- **Tab 1 — Synthetic Demo:** chọn ảnh Stanford Cars + loại occlusion + config → xem so sánh 4 ảnh (x_gt | x_occ | canny | x_hat)
- **Tab 2 — Interactive:** upload ảnh xe tùy chọn → brush mask → phục dựng → download
- Deploy: HuggingFace Spaces (ZeroGPU free tier)
- Latency estimate: ~10–20 giây/ảnh trên GPU

---

*— End of PROJECT_CONTEXT.md —*
*Nếu có thay đổi kiến trúc hoặc dataset, cập nhật file này trước khi làm việc với agent.*
