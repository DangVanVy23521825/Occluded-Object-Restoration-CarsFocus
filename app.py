"""
Streamlit demo — Car Inpainting (Tối ưu GPU).
"""
import io
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from inference import inpaint
from model_loader import load_pipelines
from segmentation import auto_detect_car_mask

st.set_page_config(page_title="Car Inpainting Demo", layout="wide")
st.title("🚗 Auto Car Inpainting - Local GPU Demo")

# Khởi tạo session_state
if "result_img" not in st.session_state:
    st.session_state.result_img = None
if "auto_mask" not in st.session_state:
    st.session_state.auto_mask = None

# Đường dẫn cứng tới file trọng số LoRA mới nhất của bạn
LORA_PATH = "adapter_model.safetensors"

# Gọi hàm load pipeline (sẽ được cache trên GPU)
pipe = load_pipelines(LORA_PATH)

# Layout giao diện
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload & Cấu hình")
    uploaded_file = st.file_uploader("Chọn ảnh gốc", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.subheader("⚙️ Cấu hình Suy luận")
    steps = st.slider("Inference Steps", 10, 50, 20)
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
    cn_scale = st.slider("ControlNet Scale", 0.0, 1.0, 0.5)
    ip_scale = st.slider("IP-Adapter Scale", 0.0, 1.0, 0.5)

    if pipe is not None:
        pipe.set_ip_adapter_scale(ip_scale)

with col2:
    st.header("2. Khu vực tạo Mask")
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        # Thay đổi kích thước hiển thị cho đỡ nặng UI
        base_w = 600
        w_pct = (base_w / float(raw_img.size[0]))
        h_size = int((float(raw_img.size[1]) * float(w_pct)))
        disp_img = raw_img.resize((base_w, h_size), Image.Resampling.LANCZOS)
        
        mode = st.radio("Chế độ tạo Mask:", ["🖌️ Vẽ bằng Brush", "🎯 Tự động phát hiện (YOLO)"], horizontal=True)
        
        final_mask = None

        if mode == "🎯 Tự động phát hiện (YOLO)":
            if st.button("Tự động khoanh vùng xe", type="primary"):
                with st.spinner("YOLO đang quét..."):
                    mask = auto_detect_car_mask(disp_img)
                    if mask:
                        st.session_state.auto_mask = mask
                        st.success("Đã tìm thấy vật thể!")
                    else:
                        st.warning("Không tìm thấy xe nào trong ảnh.")
            
            if st.session_state.auto_mask:
                st.image(st.session_state.auto_mask, caption="Mask tự động", width=base_w)
                final_mask = st.session_state.auto_mask.resize(raw_img.size, Image.Resampling.NEAREST)

        else:
            st.write("Vẽ vùng cần inpaint:")
            stroke_width = st.slider("Kích cỡ cọ", 5, 50, 25)
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.4)",  # <-- ĐỔI DÒNG NÀY (Đỏ, độ mờ 40%)
                stroke_width=stroke_width,
                stroke_color="rgba(255, 0, 0, 1)",  # Có thể đổi stroke sang đỏ luôn cho ngầu
                background_image=disp_img,
                height=h_size,
                width=base_w,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None:
                mask_np = canvas_result.image_data[:, :, 3] # Lấy alpha channel
                mask_disp = Image.fromarray(mask_np).convert("L")
                final_mask = mask_disp.resize(raw_img.size, Image.Resampling.NEAREST)

st.markdown("---")
st.header("3. Kết quả")

if st.button("🚀 Thực thi Inpainting", type="primary", use_container_width=True):
    if uploaded_file is None or final_mask is None or np.sum(np.array(final_mask)) == 0:
        st.error("Vui lòng tải ảnh và tạo Mask trước!")
    else:
        with st.spinner("⏳ Đang chạy Stable Diffusion + ControlNet..."):
            # Gọi hàm inpaint từ inference.py
            result_img, msg, lp, ss = inpaint(
                pipe, raw_img, final_mask,
                steps=steps, guidance=guidance, cn_scale=cn_scale
            )
            st.session_state.result_img = result_img
            st.success(msg)

if st.session_state.result_img:
    res, m1, m2 = st.columns([2, 1, 1])
    res.image(st.session_state.result_img, caption="Ảnh Khôi phục", use_column_width=True)
    
    # Download Button
    buf = io.BytesIO()
    st.session_state.result_img.save(buf, format="PNG")
    m1.download_button(
        label="⬇️ Tải ảnh xuống",
        data=buf.getvalue(),
        file_name="inpainted_car.png",
        mime="image/png"
    )