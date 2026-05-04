"""
Load four Stable Diffusion inpainting pipelines (cached once per Streamlit session).
Memory Optimized with Shared Components & Native LoRA.
"""

from __future__ import annotations

import lpips
import streamlit as st
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
)
# Đã xoá dòng: from peft import PeftModel


def _device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return device, dtype


@st.cache_resource(show_spinner="Loading models (Memory Optimized)...")
def load_pipelines(lora_path: str) -> dict:
    device, dtype = _device_dtype()

    # ==========================================
    # 1. LOAD SHARED COMPONENTS (TIẾT KIỆM RAM)
    # ==========================================
    temp_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    shared_vae = temp_pipe.vae
    shared_text_encoder = temp_pipe.text_encoder
    shared_tokenizer = temp_pipe.tokenizer
    shared_scheduler = DPMSolverMultistepScheduler.from_config(temp_pipe.scheduler.config)
    shared_feature_extractor = temp_pipe.feature_extractor
    
    del temp_pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    shared_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # ==========================================
    # 2. BUILD PIPELINE BASE
    # ==========================================
    pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        vae=shared_vae,
        text_encoder=shared_text_encoder,
        tokenizer=shared_tokenizer,
        scheduler=shared_scheduler,
        feature_extractor=shared_feature_extractor,
    )
    # pipe_base.enable_attention_slicing("auto")
    pipe_base.vae.enable_slicing()
    pipe_base = pipe_base.to(device)

    # ==========================================
    # 3. BUILD PIPELINE LORA BASE
    # ==========================================
    lora_pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        vae=shared_vae,
        text_encoder=shared_text_encoder,
        tokenizer=shared_tokenizer,
        scheduler=shared_scheduler,
        feature_extractor=shared_feature_extractor,
    )
    
    # [ĐÃ SỬA] Dùng hàm native thay cho thư viện Peft
    lora_pipe_base.load_lora_weights(lora_path)
    lora_pipe_base.fuse_lora()
    
    # lora_pipe_base.enable_attention_slicing("auto")
    lora_pipe_base.vae.enable_slicing()
    lora_pipe_base = lora_pipe_base.to(device)

    # ==========================================
    # 4. BUILD PIPELINE CONTROLNET
    # ==========================================
    pipe_cn = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=shared_controlnet,
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        vae=shared_vae,
        text_encoder=shared_text_encoder,
        tokenizer=shared_tokenizer,
        scheduler=shared_scheduler,
        feature_extractor=shared_feature_extractor,
    )
    pipe_cn.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin", low_cpu_mem_usage=True
    )
    # pipe_cn.enable_attention_slicing("auto")
    pipe_cn.vae.enable_slicing()
    pipe_cn = pipe_cn.to(device)

    # ==========================================
    # 5. BUILD PIPELINE CONTROLNET + LORA
    # ==========================================
    lora_pipe_cn = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=shared_controlnet,
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        vae=shared_vae,
        text_encoder=shared_text_encoder,
        tokenizer=shared_tokenizer,
        scheduler=shared_scheduler,
        feature_extractor=shared_feature_extractor,
    )
    
    lora_pipe_cn.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin", low_cpu_mem_usage=True
    )
    
    # [ĐÃ SỬA] Dùng hàm native của Diffusers giải quyết xung đột IP-Adapter
    lora_pipe_cn.load_lora_weights(lora_path)
    lora_pipe_cn.fuse_lora()
    
    # lora_pipe_cn.enable_attention_slicing("auto")
    lora_pipe_cn.vae.enable_slicing()
    lora_pipe_cn = lora_pipe_cn.to(device)

    # ==========================================
    # 6. LOAD LPIPS MODEL
    # ==========================================
    lp_model = lpips.LPIPS(net="alex", spatial=True).to(device)
    lp_model.eval()

    return {
        "device": device,
        "dtype": dtype,
        "pipe_base": pipe_base,
        "lora_pipe_base": lora_pipe_base,
        "pipe_cn": pipe_cn,
        "lora_pipe_cn": lora_pipe_cn,
        "lpips_model": lp_model,
    }