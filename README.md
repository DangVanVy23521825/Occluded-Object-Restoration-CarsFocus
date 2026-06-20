<h1 align="center">🚗 Occluded Object Restoration — Cars Focus</h1>

<p align="center">
  <b>Image inpainting for occluded vehicles using Stable Diffusion + ControlNet + IP-Adapter + LoRA fine-tuning</b><br/>
  <i>Reconstructing missing regions with global structural and textural coherence</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Stable%20Diffusion-Inpainting-purple?logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/ControlNet-SD1.5%20Canny-orange"/>
  <img src="https://img.shields.io/badge/IP--Adapter-Plus%20SD1.5-green"/>
  <img src="https://img.shields.io/badge/LoRA-PEFT%20Fine--tuned-red"/>
  <img src="https://img.shields.io/badge/YOLOv8-Auto%20Segmentation-darkblue?logo=ultralytics"/>
  <img src="https://img.shields.io/badge/Streamlit-Demo%20App-ff4b4b?logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Course-CS331%20Computer%20Vision-lightgrey"/>
</p>

---

## Overview

**Occluded Object Restoration — Cars Focus** is a computer vision project for **CS331 (Computer Vision)** at UIT–VNU Ho Chi Minh City. The system restores occluded regions of vehicles in images using a full generative inpainting pipeline built on top of Stable Diffusion 1.5.

The pipeline combines **three complementary conditioning mechanisms**:

- **ControlNet (Canny)** — preserves structural edges of the surrounding context to guide structural coherence during generation.
- **IP-Adapter Plus** — leverages the visible car region as a visual prompt to ensure textural and style consistency.
- **LoRA fine-tuning (PEFT)** — domain-adapted weights trained on car images to improve generation quality specific to vehicle restoration.

A **Streamlit interactive demo** allows users to upload images, draw or auto-detect occlusion masks, configure the inference pipeline, and download restored results.

---

## Architecture

```
Input Image + Mask
        │
        ▼
┌─────────────────────────────────────────────┐
│             Pre-processing                  │
│  • Letterbox resize to 512×512              │
│  • Canny edge extraction (masked region     │
│    zeroed out) → ControlNet conditioning    │
│  • Visible patch extraction → IP-Adapter    │
│    visual prompt                            │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│     StableDiffusionControlNetInpaint        │
│  ┌───────────────────────────────────────┐  │
│  │  Base: runwayml/stable-diffusion-     │  │
│  │         inpainting                    │  │
│  │  + ControlNet: SD1.5 Canny            │  │
│  │  + IP-Adapter: ip-adapter-plus_sd15   │  │
│  │  + LoRA: PEFT adapter (r=8, fused)    │  │
│  │  Scheduler: DPMSolver++ (Karras)      │  │
│  └───────────────────────────────────────┘  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│             Post-processing                 │
│  • Unpad & resize back to original dims     │
│  • Gaussian-blurred mask compositing        │
│    (paste generated region onto original)   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
              Restored Image
```

---

## Key Features

- **Dual mask creation modes** — freehand brush drawing via canvas, or fully automatic car detection using YOLOv8n-seg (COCO classes: car, truck, bus, motorcycle).
- **Letterbox padding strategy** — preserves aspect ratio when resizing to 512×512 for inference, then crops back to original resolution.
- **Masked Canny conditioning** — edge map is computed only on unmasked regions, preventing the model from "seeing" the occluded area through structural hints.
- **Visible patch IP-Adapter** — fills the masked region with the mean color of visible pixels to form a clean reference patch for style conditioning.
- **LoRA fused inference** — fine-tuned adapter weights are fused directly into the UNet for zero-overhead inference.
- **GPU memory optimization** — `enable_model_cpu_offload()` + VAE slicing for stable inference on consumer GPUs (~12 GB VRAM recommended).
- **Evaluation metrics** — masked SSIM and LPIPS computed via `metrics.py` for quantitative assessment of restoration quality.

---

## Tech Stack

| Component | Technology |
|---|---|
| Base Diffusion Model | `runwayml/stable-diffusion-inpainting` |
| Structural Conditioning | `lllyasviel/control_v11p_sd15_canny` (ControlNet SD1.5) |
| Visual Style Conditioning | `h94/IP-Adapter` — `ip-adapter-plus_sd15.bin` |
| Fine-tuning | LoRA / PEFT (`outputs/lora_weights/r8/best`) |
| Auto Segmentation | YOLOv8n-seg (Ultralytics) — COCO vehicle classes |
| Scheduler | DPMSolver++ with Karras sigmas |
| Demo Interface | Streamlit + streamlit-drawable-canvas |
| Image Processing | OpenCV, Pillow, NumPy |
| Evaluation | SSIM (skimage), LPIPS |
| Deep Learning Framework | PyTorch (CUDA) |

---

## Project Structure

```
Occluded-Object-Restoration-CarsFocus/
├── app.py                    # Streamlit demo application
├── inference.py              # Core inpainting pipeline (LoRA + ControlNet + IP-Adapter)
├── model_loader.py           # Pipeline loading & GPU memory management (cached)
├── segmentation.py           # YOLOv8-based auto car mask generation
├── metrics.py                # Evaluation metrics (masked SSIM, LPIPS)
├── requirements.txt          # Python dependencies
├── notebooks/                # Training & experimentation notebooks
│   └── ...
└── outputs/
    └── lora_weights/
        └── r8/
            └── best/         # Trained LoRA adapter (PEFT format)
```

---

## Getting Started

### Prerequisites

| Requirement | Recommended |
|---|---|
| Python | 3.10+ (tested on 3.12) |
| GPU VRAM | ~12 GB NVIDIA CUDA (CPU supported but slow) |
| Disk Space | ~15 GB for Hugging Face model cache |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/DangVanVy23521825/Occluded-Object-Restoration-CarsFocus.git
cd Occluded-Object-Restoration-CarsFocus

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### LoRA Weights Setup

The app requires trained LoRA weights in PEFT format. Place them at:

```
outputs/lora_weights/r8/best/   ← adapter PEFT files + config
```

You can override the default path via environment variable or Streamlit secrets:

```bash
# Option A: Environment variable
export LORA_WEIGHTS_PATH="/absolute/path/to/lora/best"

# Option B: Streamlit secrets (recommended)
# Create .streamlit/secrets.toml at repo root:
# LORA_PATH = "outputs/lora_weights/r8/best"
```

> **Note:** Do not commit `secrets.toml` if it contains sensitive paths — add it to `.gitignore`.

### Run the Demo

Always run from the **repository root** to ensure correct path resolution:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Demo Usage

1. **Upload** a vehicle image (JPG/PNG).
2. **Create a mask** — choose between:
   - 🖌️ **Brush mode** — draw freehand over occluded regions on the interactive canvas.
   - 🎯 **Auto-detect (YOLO)** — automatically segments vehicles using YOLOv8n-seg.
3. **Configure inference** — adjust Steps (10–50), Guidance Scale, ControlNet Scale, and IP-Adapter Scale via sliders.
4. **Run** — click **Restore** and wait for Stable Diffusion to reconstruct the masked region.
5. **Download** — save the restored PNG via the download button.

---

## Inference Pipeline Details

The `inpaint()` function in `inference.py` executes the following steps:

1. Letterbox-pad image and mask to 512×512.
2. Extract a masked Canny edge map — edges inside the mask region are zeroed to prevent information leakage.
3. Extract a visible-region patch as the IP-Adapter reference image.
4. Run `StableDiffusionControlNetInpaintPipeline` with DPMSolver++ (Karras).
5. Unpad and resize result back to the original image resolution.
6. Composite the result onto the original via Gaussian-blurred mask feathering.

```python
from inference import inpaint
from model_loader import load_pipelines
from PIL import Image

pipe = load_pipelines("outputs/lora_weights/r8/best/adapter_model.safetensors")
result, msg, canny_info, canny_img = inpaint(
    pipe, image_pil, mask_pil,
    steps=20, guidance=7.5, cn_scale=0.7, ip_scale=0.5, seed=42
)
```

---

## Evaluation

Restoration quality is measured with two metrics computed **only on masked regions**:

| Metric | Description |
|---|---|
| **Masked SSIM** | Structural Similarity Index restricted to the inpainted area |
| **LPIPS** | Learned Perceptual Image Patch Similarity — perceptual quality |

Run evaluation via `metrics.py` on pairs of original and restored images.

---

## System Requirements & Troubleshooting

| Symptom | Solution |
|---|---|
| `No module named 'streamlit_drawable_canvas'` | `pip install streamlit-drawable-canvas` |
| `No module named 'peft'` or missing packages | Re-run `pip install -r requirements.txt` in the correct venv |
| `StreamlitSecretNotFoundError` | Create `.streamlit/secrets.toml` as described above |
| `No space left on device` during model download | Free up disk space or redirect cache: `export HF_HOME=/path/to/large/disk` |
| Slow first startup | Expected — `from_pretrained` downloads from Hugging Face; subsequent runs use local cache |
| `CUDA OOM` during inference | Reduce inference steps or enable `pipe.vae.enable_tiling()` |

---

<p align="center">
  <i>CS331 Computer Vision — UIT VNU-HCM</i>
</p>
