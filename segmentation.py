"""
Auto-segmentation using YOLOv8-seg (Ultralytics).
Phát hiện xe cộ và tự động tạo Mask đen trắng.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Tải model YOLOv8n-seg cực nhẹ (khoảng vài MB) để chạy mượt trên local
MODEL_NAME = "yolov8n-seg.pt"

def auto_detect_car_mask(image_pil: Image.Image) -> Image.Image | None:
    """
    Sử dụng YOLO để tìm và tạo mask cho xe cộ (car/truck/bus).
    Trả về PIL Image (L mode) chứa Mask đen/trắng, hoặc None nếu không tìm thấy.
    """
    model = YOLO(MODEL_NAME)
    
    # Chạy inference trên ảnh
    results = model.predict(image_pil, conf=0.25, verbose=False)
    result = results[0]
    
    if result.masks is None:
        return None
    
    # Lấy các class id đã detect (car=2, motorcycle=3, bus=5, truck=7 theo COCO)
    vehicle_classes = [2, 3, 5, 7]
    
    # Tìm vùng mask tổng hợp của tất cả các xe
    combined_mask = np.zeros(image_pil.size[::-1], dtype=np.uint8) # (H, W)
    masks_data = result.masks.data.cpu().numpy()
    boxes_cls = result.boxes.cls.cpu().numpy()
    
    has_vehicle = False
    for i, cls in enumerate(boxes_cls):
        if int(cls) in vehicle_classes:
            has_vehicle = True
            # YOLO mask (H,W) dạng float 0.0-1.0, cần scale lên đúng kích thước ảnh gốc
            mask_resized = torch.nn.functional.interpolate(
                torch.tensor(masks_data[i]).unsqueeze(0).unsqueeze(0),
                size=(image_pil.height, image_pil.width),
                mode="bilinear"
            ).squeeze().numpy()
            
            combined_mask[mask_resized > 0.5] = 255
            
    if not has_vehicle:
        return None
        
    return Image.fromarray(combined_mask, mode="L")