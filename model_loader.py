"""
Load 1 pipeline Full Stack (LoRA + ControlNet + IP-Adapter) hoàn chỉnh trên GPU.
"""
from __future__ import annotations
import gc
import torch
import streamlit as st
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
)

BASE_MODEL = "runwayml/stable-diffusion-inpainting"
CN_MODEL   = "lllyasviel/control_v11p_sd15_canny"
IPA_REPO   = "h94/IP-Adapter"
IPA_WEIGHT = "ip-adapter-plus_sd15.bin"

def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

@st.cache_resource(show_spinner="⏳ Đang tải mô hình lên GPU (~2-3 phút)...")
def load_pipelines(lora_safetensors_path: str) -> StableDiffusionControlNetInpaintPipeline:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    # 1. Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        CN_MODEL, torch_dtype=dtype, use_safetensors=True
    )
    _free()

    # 2. Load Base Pipeline Inpainting
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    _free()

    # 3. Load IP-Adapter
    pipe.load_ip_adapter(
        IPA_REPO,
        subfolder="models",
        weight_name=IPA_WEIGHT,
    )
    # Tùy chỉnh mức độ ảnh hưởng của IP-Adapter (0.0 -> 1.0)
    pipe.set_ip_adapter_scale(0.5) 
    _free()

    # 4. Load LoRA (native Diffusers)
    try:
        pipe.load_lora_weights(lora_safetensors_path)
        pipe.fuse_lora()
    except Exception as e:
        print(f"⚠️ Không thể load/fuse LoRA: {e}")
    _free()

    # 5. Tối ưu Memory
    pipe.vae.enable_slicing()
    if device.type == "cuda":
        pipe.enable_model_cpu_offload() # Tự động swap memory thông minh để tối ưu GPU
        
    return pipe