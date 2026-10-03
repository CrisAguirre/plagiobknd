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

# --- Detección ampliada de fechas (reformulación: reemplazar in-situ) ---
import re as _re

_MESES = r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
_ABBR = r"ene|feb|mar|abr|may|jun|jul|ago|sep|set|oct|nov|dic"

P_NUM = _re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
P_ISO = _re.compile(r"\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b")
P_ABBR = _re.compile(rf"\b\d{{1,2}}[\s\-/\.]*(?:{_ABBR})[\s\-/\.]*\d{{2,4}}\b", _re.IGNORECASE)
P_ABBR2 = _re.compile(rf"\b\d{{1,2}}[\s\-/\.]*(?:{_ABBR})[\s\-/\.]*\d{{1,2}}[\s\-/\.]+\d{{2,4}}\b", _re.IGNORECASE)
P_LONG = _re.compile(rf"\b\d{{1,2}}\s+de\s+(?:{_MESES})\s+(?:de\s+)?\d{{2,4}}\b", _re.IGNORECASE)

_SINGLE_PATS = (P_NUM, P_ISO, P_ABBR, P_ABBR2)
# Multi-token SOLO para fechas que requieren varias palabras ('12 de marzo de 2026',
# '12 mar 2026'). NO incluir P_NUM/P_ISO aquí: si no, 'EL DIA 22/03/2023' generaría
# un box gigante que borraría contexto. Esos ya se detectan como single-token.
_MULTI_PATS = (P_LONG, P_ABBR, P_ABBR2)


def _is_date_token(t: str) -> bool:
    return any(p.search(t) for p in _SINGLE_PATS)


def _is_date_phrase(s: str) -> bool:
    return any(p.search(s) for p in _MULTI_PATS)


def _ensure_tesseract():
    """Local Windows: usa Tesseract si no esta en PATH. En Docker ya esta en PATH."""
    try:
        import shutil
        import pytesseract
        if shutil.which("tesseract") is None:
            for cand in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                         r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                if os.path.exists(cand):
                    pytesseract.pytesseract.tesseract_cmd = cand
                    break
    except Exception:
        pass


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Escaneados con fondo gris: binariza con umbral adaptativo por página
    (media - 0.8*std, clamped 140..200). Sin resize: mismas coords.
    """
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    m, s = float(gray.mean()), float(gray.std())
    thv = int(m - 0.8 * s)
    thv = max(140, min(200, thv))
    _, binimg = cv2.threshold(gray, thv, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binimg)


def detect_date_boxes(img: Image.Image, min_conf: float = 30.0):
    """
    Detecta TODAS las fechas visibles en la página.
    Devuelve [{text, box:{x0,y0,x1,y1 0-1000}, conf 0-1}].
    - 1 token: dd/mm/aaaa, aaaa-mm-dd, 12-mar-2026
    - multi-token 3..5: '12 de marzo de 2026'
    Preprocesa (binariza) copia para OCR; coords referidas al original.
    """
    import pytesseract
    from pytesseract import Output
    _ensure_tesseract()
    ocr_img = _preprocess_for_ocr(img)
    d = pytesseract.image_to_data(ocr_img, lang="spa+eng", config="--oem 1 --psm 6",
                                  output_type=Output.DICT)
    W, H = img.size
    n = len(d["text"])
    words = []
    for i in range(n):
        t = (d["text"][i] or "").strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        words.append({
            "text": t, "conf": conf,
            "l": int(d["left"][i]), "t": int(d["top"][i]),
            "w": int(d["width"][i]), "h": int(d["height"][i]),
        })

    found = []

    def to1000(l, t, w, h, conf, text):
        return {
            "text": text,
            "box": {
                "x0": l / W * 1000, "y0": t / H * 1000,
                "x1": (l + w) / W * 1000, "y1": (t + h) / H * 1000,
            },
            "conf": max(0.0, min(1.0, conf / 100.0)),
        }

    # 1) single-token
    for w in words:
        if w["conf"] < min_conf:
            continue
        if _is_date_token(w["text"]):
            found.append(to1000(w["l"], w["t"], w["w"], w["h"], w["conf"], w["text"]))

    # 2) multi-token ventanas 3..5 (para '12 de marzo de 2026')
    for size in (5, 4, 3):
        for i in range(len(words) - size + 1):
            win = words[i:i + size]
            if any(x["conf"] < min_conf for x in win):
                continue
            # Si ya hay fecha single-token dentro, quedarse con el box ajustado
            if any(_is_date_token(x["text"]) for x in win):
                continue
            phrase = " ".join(x["text"] for x in win)
            if not _is_date_phrase(phrase):
                continue
            l = min(x["l"] for x in win)
            t = min(x["t"] for x in win)
            r = max(x["l"] + x["w"] for x in win)
            b = max(x["t"] + x["h"] for x in win)
            avg = sum(x["conf"] for x in win) / len(win)
            cand = to1000(l, t, r - l, b - t, avg, phrase)
            # evita duplicados que solapen >50% con uno ya hallado
            dup = False
            for f in found:
                a, o = f["box"], cand["box"]
                ix0, iy0 = max(a["x0"], o["x0"]), max(a["y0"], o["y0"])
                ix1, iy1 = min(a["x1"], o["x1"]), min(a["y1"], o["y1"])
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    area = (o["x1"] - o["x0"]) * (o["y1"] - o["y0"])
                    if area and inter / area > 0.5:
                        dup = True
                        break
            if not dup:
                found.append(cand)
    return found


def _load_font(size: int, bold: bool = False):
    # Regular primero: igual ancho que el original escaneado (el bold ensancha
    # y obligaba a encoger la fuente -> se veia pequena). Bold solo si se pide.
    regular = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
               "LiberationSerif-Regular.ttf"]
    bold_names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] + regular
    for name in (bold_names if bold else regular):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _sample_ink_color(img: Image.Image, x0: int, y0: int, x1: int, y1: int):
    """Toma el color real del texto original (mediana de pixeles oscuros)."""
    try:
        crop = img.crop((max(0, x0), max(0, y0), x1, y1)).convert("RGB")
        import numpy as _np
        a = _np.array(crop).reshape(-1, 3)
        gray = a.mean(axis=1)
        dark = a[gray < _np.percentile(gray, 15)]
        if len(dark) == 0:
            return (25, 25, 25)
        med = _np.median(dark, axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))
    except Exception:
        return (25, 25, 25)


def _analyze_orig(img: Image.Image, x0: int, y0: int, x1: int, y1: int):
    """
    Mide el texto ORIGINAL antes de borrarlo:
    - text_h: alto real de tinta (sin padding Tesseract) -> objetivo de altura 1:1
    - bold: por grosor de trazo (erosion ratio), no por posicion
    - ink: color mediana de pixeles oscuros
    - bg_sigma: grano del fondo vecino (anillo 12px) para igualar nitidez
    Fallbacks seguros si algo falla.
    """
    box_w = max(1, x1 - x0); box_h = max(1, y1 - y0)
    try:
        crop = img.crop((max(0, x0), max(0, y0), x1, y1)).convert("RGB")
        g = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
        _, bin_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = bin_inv.shape
        # filas con tinta (ignora motas: >2% del ancho)
        rows = np.where((bin_inv > 0).sum(axis=1) > max(1, w * 0.02))[0]
        text_h = int(rows[-1] - rows[0] + 1) if len(rows) else int(box_h * 0.7)
        text_h = max(6, min(box_h, text_h))
        # grosor: cuanto sobrevive a erosion 3x3 (bold retiene mas)
        eroded = cv2.erode(bin_inv, np.ones((3, 3), np.uint8), iterations=1)
        s_bin, s_ero = float(bin_inv.sum()), float(eroded.sum())
        ratio = (s_ero / s_bin) if s_bin > 0 else 0.0
        bold = ratio > 0.38
        # tinta: mediana de pixeles oscuros
        a = np.array(crop).reshape(-1, 3)
        gray = a.mean(axis=1)
        dark = a[gray < np.percentile(gray, 15)]
        ink = tuple(int(v) for v in np.median(dark, axis=0)) if len(dark) else (25, 25, 25)
    except Exception:
        text_h, bold, ink = int(box_h * 0.7), False, (25, 25, 25)
    try:
        W, H = img.size
        rx0, ry0 = max(0, x0 - 12), max(0, y0 - 12)
        rx1, ry1 = min(W, x1 + 12), min(H, y1 + 12)
        ring = np.array(img.crop((rx0, ry0, rx1, ry1)).convert("L"), dtype=np.float32)
        m = np.zeros_like(ring, dtype=bool)
        m[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = True
        outside = ring[~m]
        # solo papel (pixeles claros): excluye texto/lineas vecinas que inflaban sigma
        if outside.size > 20:
            thr = float(np.median(outside))
            paper = outside[outside >= thr]
            bg_sigma = float(paper.std()) if paper.size > 20 else float(outside.std())
        else:
            bg_sigma = 2.5
        bg_sigma = max(1.2, min(5.0, bg_sigma))
    except Exception:
        bg_sigma = 2.5
    return {"text_h": text_h, "bold": bold, "ink": ink, "bg_sigma": bg_sigma}


def _scan_match(img: Image.Image, x0: int, y0: int, x1: int, y1: int, sigma: float):
    """Iguala nitidez/grano del parche con el escaneo: blur leve + ruido del fondo."""
    try:
        from PIL import ImageFilter as _IF
        region = img.crop((x0, y0, x1, y1))
        region = region.filter(_IF.GaussianBlur(0.45))
        arr = np.array(region).astype(np.int16)
        noise = np.random.normal(0, sigma * 0.7, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(arr), (x0, y0))
    except Exception:
        pass
    return img


def _inpaint_box(img: Image.Image, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    """Elimina el contenido del box reconstruyendo el fondo (inpaint TELEA)."""
    W, H = img.size
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    mask = np.zeros((H, W), dtype=np.uint8)
    pad = max(2, int((x1 - x0) * 0.03))
    cv2.rectangle(mask, (max(0, x0 - pad), max(0, y0 - pad)),
                  (min(W, x1 + pad), min(H, y1 + pad)), 255, -1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(cv_img, mask, 4, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))


def _draw_fitted_text(img: Image.Image, x0: int, y0: int, x1: int, y1: int, text: str,
                        ink=None, bold: bool = False, target_h: int = 0):
    """
    Escribe `text` centrado, adaptado al ORIGINAL medido y nítido:
    - altura objetivo = text_h del original (sin padding Tesseract), tolerancia -1px.
      Si no se mide (0), fallback alto<=82% del box.
    - ancho <=100% del box.
    - supermuestreo 4x + LANCZOS (bordes suaves, sin pixelado).
    """
    from PIL import Image as _PI
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    if ink is None:
        ink = (25, 25, 25)
    want_h = target_h if target_h and target_h > 0 else int(box_h * 0.82)
    want_h = max(6, min(box_h, want_h))
    SS = 4
    probe = ImageDraw.Draw(img)
    size = max(8, int(want_h * 1.45))
    font = _load_font(size, bold=bold)
    tw = th = 0
    while size > 8:
        font = _load_font(size, bold=bold)
        try:
            bb = probe.textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            try:
                tw = int(font.getlength(text)); th = size
            except Exception:
                break
        if tw <= box_w * 1.0 and th <= want_h and th >= want_h - 1:
            break
        if tw <= box_w * 1.0 and th <= want_h:
            break
        size -= 1
    # recorte ya con inpaint -> ampliar, dibujar grande, reducir (bordes suaves)
    try:
        crop_big = img.crop((x0, y0, x1, y1)).resize((box_w * SS, box_h * SS), _PI.BICUBIC)
    except Exception:
        crop_big = _PI.new("RGB", (box_w * SS, box_h * SS), (255, 255, 255))
    d_big = ImageDraw.Draw(crop_big)
    font_big = _load_font(size * SS, bold=bold)
    try:
        bb = d_big.textbbox((0, 0), text, font=font_big)
        twb, thb = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        twb, thb = box_w * SS // 2, box_h * SS // 2
        bb = (0, 0, twb, thb)
    tx = (box_w * SS - twb) // 2 - bb[0]
    ty = (box_h * SS - thb) // 2 - bb[1]
    d_big.text((tx, ty), text, fill=ink, font=font_big)
    try:
        small = crop_big.resize((box_w, box_h), _PI.LANCZOS)
    except Exception:
        small = crop_big.resize((box_w, box_h))
    img.paste(small, (x0, y0))
    return img

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
    3. overlay texto nuevo auto-ajustado y centrado
    4. re-empaqueta solo esa página a PDF
    Devuelve PDF de 1 página. Para documento completo usar replace_dates_in_pdf().
    """
    img = render_page(pdf_bytes, page_index)
    W, H = img.size
    x0, y0, x1, y1 = _box_to_pixels(box, W, H)

    ana = _analyze_orig(img, x0, y0, x1, y1)
    img = _inpaint_box(img, x0, y0, x1, y1)
    img = _draw_fitted_text(img, x0, y0, x1, y1, text, ink=ana["ink"],
                            bold=ana["bold"], target_h=ana["text_h"])
    img = _scan_match(img, x0, y0, x1, y1, ana["bg_sigma"])

    # --- a PDF sin recompresión extra ---
    import img2pdf
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pdf_out = img2pdf.convert(buf.getvalue())
    return pdf_out


def replace_dates_in_pdf(pdf_bytes: bytes, new_text: str, dpi: int = 200):
    """
    Reformulación principal: detecta TODAS las fechas en TODAS las páginas,
    las elimina con inpaint y sobrepone `new_text` in-situ.
    Devuelve (pdf_out: bytes, report: {total, pages, per_page:[{page_index,count,boxes}]})
    El PDF conserva el mismo nº de páginas; las no tocadas se re-empaquetan tal cual.
    """
    new_text = (new_text or "").strip()
    if not new_text:
        raise ValueError("new_text vacío")
    dpi = max(100, min(300, int(dpi)))

    pdf = pdfium.PdfDocument(pdf_bytes)
    n = len(pdf)
    if n < 1:
        raise ValueError("PDF sin páginas")

    png_list: list[bytes] = []
    per_page = []
    total = 0
    for i in range(n):
        img = render_page(pdf_bytes, i, dpi=dpi)
        W, H = img.size
        try:
            boxes = detect_date_boxes(img)
        except Exception as e:
            raise RuntimeError(f"ocr no disponible (instala tesseract): {e}")

        for b in boxes:
            bx = b["box"]
            # dict 0-1000 -> píxeles
            x0 = max(0, int(bx["x0"] / 1000.0 * W))
            y0 = max(0, int(bx["y0"] / 1000.0 * H))
            x1 = min(W, int(bx["x1"] / 1000.0 * W))
            y1 = min(H, int(bx["y1"] / 1000.0 * H))
            if x1 <= x0 or y1 <= y0:
                continue
            ana = _analyze_orig(img, x0, y0, x1, y1)
            img = _inpaint_box(img, x0, y0, x1, y1)
            img = _draw_fitted_text(img, x0, y0, x1, y1, new_text, ink=ana["ink"],
                                    bold=ana["bold"], target_h=ana["text_h"])
            img = _scan_match(img, x0, y0, x1, y1, ana["bg_sigma"])

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_list.append(buf.getvalue())
        total += len(boxes)
        per_page.append({"page_index": i, "count": len(boxes),
                         "boxes": [{"text": b["text"], "box": b["box"]} for b in boxes]})

    import img2pdf
    pdf_out = img2pdf.convert(*png_list)
    return pdf_out, {"total": total, "pages": n, "per_page": per_page}

def images_to_pdf(png_bytes_list: list[bytes]) -> bytes:
    import img2pdf
    return img2pdf.convert(*png_bytes_list)
