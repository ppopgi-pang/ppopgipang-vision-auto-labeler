from PIL import Image, ImageDraw, ImageFont
from domain.bbox import BoundingBox
from pathlib import Path

def _load_default_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()

def crop_image_to_pil(image_path: str | Path, bbox: BoundingBox, padding: int = 0) -> Image.Image | None:
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            x1 = max(0, bbox.x1 - padding)
            y1 = max(0, bbox.y1 - padding)
            x2 = min(width, bbox.x2 + padding)
            y2 = min(height, bbox.y2 + padding)

            crop = img.crop((x1, y1, x2, y2))
            return crop.copy()
    except Exception as e:
        print(f"[BBoxUtils] Error cropping {image_path} to PIL: {e}")
        return None

def draw_bboxes(
    image_path: str | Path,
    bboxes: list[BoundingBox],
    output_path: str | Path,
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
    font_size: int = 14,
    show_confidence: bool = True,
) -> bool:
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            font = _load_default_font(font_size)
            img_width, img_height = img.size

            for bbox in bboxes:
                x1 = int(max(0, min(img_width - 1, bbox.x1)))
                y1 = int(max(0, min(img_height - 1, bbox.y1)))
                x2 = int(max(0, min(img_width - 1, bbox.x2)))
                y2 = int(max(0, min(img_height - 1, bbox.y2)))

                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

                label = bbox.label or "unknown"
                if show_confidence:
                    label = f"{label} {bbox.confidence:.2f}"

                try:
                    text_box = draw.textbbox((x1, y1), label, font=font)
                    text_width = text_box[2] - text_box[0]
                    text_height = text_box[3] - text_box[1]
                except Exception:
                    text_width, text_height = draw.textsize(label, font=font)

                text_x1 = x1
                text_y1 = y1 - text_height - 4
                if text_y1 < 0:
                    text_y1 = y1 + 2

                text_bg = [text_x1, text_y1, text_x1 + text_width + 4, text_y1 + text_height + 4]
                draw.rectangle(text_bg, fill=color)
                draw.text((text_x1 + 2, text_y1 + 2), label, fill=(0, 0, 0), font=font)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)

        return True
    except Exception as e:
        print(f"[BBoxUtils] Error drawing bboxes for {image_path}: {e}")
        return False

def crop_image(image_path: str | Path, bbox: BoundingBox, output_path: str | Path, padding: int = 0):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            x1 = max(0, bbox.x1 - padding)
            y1 = max(0, bbox.y1 - padding)
            x2 = min(width, bbox.x2 + padding)
            y2 = min(height, bbox.y2 + padding)
            
            crop = img.crop((x1, y1, x2, y2))
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path)
            
            return True
            
    except Exception as e:
        print(f"[BBoxUtils] Error cropping {image_path}: {e}")
        return False
