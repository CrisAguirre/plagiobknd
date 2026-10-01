"""
Motor profesional: PDF escaneado -> imagen -> inpaint + overlay -> PDF.
Usa pypdfium2 (BSD, sin AGPL) + OpenCV + Pillow + img2pdf/pikepdf.
"""
import io
import os
import pypdfium2 as pdfium
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = int(os.getenv("DPI_RENDER", "300"))

def render_page(pdf_bytes: bytes, page_index: int, dpi: int = DPI) -> Image.Image:
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[page_index]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil()
    return pil.convert("RGB")

def _box_to_pixels(box, W: int, H: int):
    x0 = int(box.x0 / 1000.0 * W)
    y0 = int(box.y0 / 1000.0 * H)
    x1 = int(box.x1 / 1000.0 * W)
    y1 = int(box.y1 / 1000.0 * H)
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

def apply_edit(pdf_bytes: bytes, page_index: int, box, text: str) -> bytes:
    """
    1. render 300 DPI
    2. inpaint zona (borra fecha vieja reconstruyendo fondo)
    3. overlay texto nuevo con fuente similar
    4. re-empaqueta solo esa página a PDF (multi-página: repetir por página en front o batch)
    Devuelve PDF de 1 página. Para multi-página usar images_to_pdf().
    """
    img = render_page(pdf_bytes, page_index)
    W, H = img.size
    x0, y0, x1, y1 = _box_to_pixels(box, W, H)

    # --- inpaint ---
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    mask = np.zeros((H, W), dtype=np.uint8)
    pad = max(2, int((x1 - x0) * 0.03))
    cv2.rectangle(mask, (max(0, x0 - pad), max(0, y0 - pad)), (min(W, x1 + pad), min(H, y1 + pad)), 255, -1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(cv_img, mask, 4, cv2.INPAINT_TELEA)
    img = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

    # --- overlay texto ---
    draw = ImageDraw.Draw(img)
    box_h = max(1, y1 - y0)
    # Tamaño proporcional, intentar fuentes comunes
    size = max(8, int(box_h * 0.8))
    font = None
    for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSerif-Regular.ttf"]:
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Samplear color oscuro cercano (texto original) — simplificado a negro
    # TODO: k-means sobre borde de box para matching exacto
    draw.text((x0, y0), text, fill=(20, 20, 20), font=font)

    # --- a PDF sin recompresión extra ---
    import img2pdf
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pdf_out = img2pdf.convert(buf.getvalue())
    return pdf_out

def images_to_pdf(png_bytes_list: list[bytes]) -> bytes:
    import img2pdf
    return img2pdf.convert(png_bytes_list)
