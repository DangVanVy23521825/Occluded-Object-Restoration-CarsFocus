from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_box(draw, rect, text, fill, outline, text_font):
    draw.rounded_rectangle(rect, radius=16, fill=fill, outline=outline, width=3)
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    lines = text.split("\n")
    line_h = text_font.getbbox("Ag")[3] + 2
    y = cy - (len(lines) * line_h) / 2
    for line in lines:
        w = text_font.getbbox(line)[2]
        draw.text((cx - w / 2, y), line, fill="#111827", font=text_font)
        y += line_h


def right_mid(rect):
    x1, y1, x2, y2 = rect
    return (x2, (y1 + y2) // 2)


def left_mid(rect):
    x1, y1, _, y2 = rect
    return (x1, (y1 + y2) // 2)


def draw_arrow(draw, p1, p2, color="#111827", width=4):
    draw.line([p1, p2], fill=color, width=width)
    hx, hy = p2
    draw.polygon([(hx, hy), (hx - 14, hy - 8), (hx - 14, hy + 8)], fill=color)


def draw_label(draw, mid, text, font, dy=-24, color="#374151"):
    tw = font.getbbox(text)[2]
    draw.text((mid[0] - tw / 2, mid[1] + dy), text, fill=color, font=font)


def main():
    w, h = 2200, 1180
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    title_font = _load_font(46, bold=True)
    sub_font = _load_font(24, bold=False)
    box_font = _load_font(22, bold=True)
    note_font = _load_font(19, bold=False)

    draw.text((470, 28), "Modernized Vehicle Occlusion Recovery Pipeline", fill="#111827", font=title_font)
    draw.text(
        (300, 90),
        "Based on notebooks/full-eval.ipynb (SD1.5 + ControlNet Canny + IP-Adapter + Evaluation)",
        fill="#374151",
        font=sub_font,
    )

    boxes = {
        "input": (80, 500, 390, 650),
        "mask": (470, 500, 760, 650),
        "canny": (470, 240, 840, 390),
        "controlnet": (900, 240, 1230, 390),
        "visible": (470, 760, 840, 910),
        "ip": (900, 760, 1230, 910),
        "sd": (1280, 480, 1710, 670),
        "pred": (1780, 500, 2110, 650),
        "grid": (1780, 240, 2110, 390),
        "eval": (1780, 760, 2110, 940),
    }

    draw_box(draw, boxes["input"], "Input x_occ\n(Occluded Vehicle)", "#dbeafe", "#1d4ed8", box_font)
    draw_box(draw, boxes["mask"], "Occlusion Mask\n(mask)", "#e0f2fe", "#0369a1", box_font)
    draw_box(draw, boxes["canny"], "extract_canny_masked\nfrom x_occ + mask", "#fef3c7", "#a16207", box_font)
    draw_box(draw, boxes["controlnet"], "ControlNet\n(Canny)", "#fde68a", "#92400e", box_font)
    draw_box(draw, boxes["visible"], "extract_visible_patch\n(visible texture ref)", "#fee2e2", "#be123c", box_font)
    draw_box(draw, boxes["ip"], "IP-Adapter\n(image prompt)", "#fecaca", "#9f1239", box_font)
    draw_box(draw, boxes["sd"], "SD1.5 Inpainting\n+ Prompt / Neg Prompt", "#dcfce7", "#15803d", box_font)
    draw_box(draw, boxes["pred"], "Recovered Image\n(x_pred)", "#bbf7d0", "#166534", box_font)
    draw_box(draw, boxes["grid"], "Grid Search\n(cn_scale, ip_scale)", "#ede9fe", "#6d28d9", box_font)
    draw_box(draw, boxes["eval"], "Evaluation Report\nPSNR | SSIM | LPIPS | FID", "#ede9fe", "#5b21b6", box_font)

    # Main arrows
    draw_arrow(draw, right_mid(boxes["input"]), left_mid(boxes["mask"]), "#0369a1")
    draw_arrow(draw, right_mid(boxes["mask"]), left_mid(boxes["sd"]), "#166534")
    draw_arrow(draw, right_mid(boxes["sd"]), left_mid(boxes["pred"]), "#166534")
    draw_label(draw, (585, 560), "input preprocessing", note_font, dy=-56)
    draw_label(draw, (1015, 560), "image + mask", note_font, dy=-56)
    draw_label(draw, (1745, 560), "inpaint result", note_font, dy=-56)

    # Canny branch
    draw_arrow(draw, (390, 535), (470, 300), "#92400e")
    draw_arrow(draw, (760, 535), (620, 390), "#92400e")
    draw_arrow(draw, right_mid(boxes["canny"]), left_mid(boxes["controlnet"]), "#92400e")
    draw_arrow(draw, right_mid(boxes["controlnet"]), (1280, 540), "#92400e")
    draw_label(draw, (812, 310), "masked canny", note_font, dy=-45, color="#92400e")
    draw_label(draw, (1250, 470), "structure guidance", note_font, dy=-45, color="#92400e")

    # IP branch
    draw_arrow(draw, (390, 615), (470, 830), "#9f1239")
    draw_arrow(draw, (760, 615), (620, 760), "#9f1239")
    draw_arrow(draw, right_mid(boxes["visible"]), left_mid(boxes["ip"]), "#9f1239")
    draw_arrow(draw, right_mid(boxes["ip"]), (1280, 610), "#9f1239")
    draw_label(draw, (812, 830), "visible ref", note_font, dy=-45, color="#9f1239")
    draw_label(draw, (1245, 690), "texture guidance", note_font, dy=14, color="#9f1239")

    # Evaluation branch
    draw_arrow(draw, (1940, 500), (1940, 390), "#5b21b6")
    draw_arrow(draw, (1940, 650), (1940, 760), "#5b21b6")
    draw_label(draw, (1985, 430), "predictions", note_font, dy=-22, color="#5b21b6")
    draw_label(draw, (1985, 720), "predictions", note_font, dy=-22, color="#5b21b6")

    draw.text(
        (80, 1080),
        "Reference in notebook: x_gt, mask, occlusion_ratio bins; best setting selected via cn/ip search.",
        fill="#4b5563",
        font=note_font,
    )

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "project_pipeline_full_eval.png"
    img.save(out_png)
    print(f"Saved PNG: {out_png.resolve()}")


if __name__ == "__main__":
    main()
