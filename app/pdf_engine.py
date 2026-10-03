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

DPI = int(os.getenv("DPI_RENDER", "200"))  # escaneos nativos ~200dpi: 1:1 sin remuestreo

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


def _ocr_words(img: Image.Image):
    """OCR una vez por pagina: devuelve (words[{text,conf,l,t,w,h} px], W, H)."""
    import pytesseract
    from pytesseract import Output
    _ensure_tesseract()
    ocr_img = _preprocess_for_ocr(img)
    d = pytesseract.image_to_data(ocr_img, lang="spa+eng", config="--oem 1 --psm 6",
                                  output_type=Output.DICT)
    W, H = img.size
    words = []
    for i in range(len(d["text"])):
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
    return words, W, H


def _is_registro(box, words, W: int, H: int) -> bool:
    """
    Solo fechas de REGISTRO en cabecera:
    - banda superior (centro-y < 40% de la pagina)
    - en su misma linea hay 'FECHA' o 'REGISTRO' (etiqueta a la izquierda)
    - si hay 'NAC' (FEC.NAC / nacimiento) -> NO, aunque diga fecha cerca
    - cuerpo del documento (accidente, evolucion, etc.) -> NO
    """
    cy = (box["y0"] + box["y1"]) / 2
    if cy > 400:
        return False
    bh = max(1.0, box["y1"] - box["y0"])
    line = []
    for w in words:
        if w["conf"] < 15:
            continue
        wcy = (w["t"] + w["h"] / 2) / H * 1000
        if abs(wcy - cy) > max(bh * 0.9, 14):
            continue
        wcx = (w["l"] + w["w"] / 2) / W * 1000
        if box["x0"] - 380 <= wcx <= box["x1"] + 150:
            line.append(w["text"])
    t = " ".join(line).upper()
    if "NAC" in t:
        return False
    return ("REGISTRO" in t) or ("FECHA" in t)


def _line_angle(words, box, W: int, H: int) -> float:
    """
    Inclinacion real del renglon donde vive la fecha (grados, + = baja a la derecha).
    Regresion sobre centros de palabras vecinas en banda angosta; por renglon,
    no por pagina (el escaneo/papel puede pandear distinto en cada zona).
    """
    cy = (box["y0"] + box["y1"]) / 2
    bh = max(1.0, box["y1"] - box["y0"])
    xs, ys = [], []
    for w in words:
        if w["conf"] < 15:
            continue
        wcy = (w["t"] + w["h"] / 2) / H * 1000
        if abs(wcy - cy) > max(bh * 0.5, 8):
            continue
        wcx = (w["l"] + w["w"] / 2) / W * 1000
        if box["x0"] - 380 <= wcx <= box["x1"] + 150:
            xs.append(wcx); ys.append(wcy)
    if len(xs) >= 2 and (max(xs) - min(xs)) > 120:
        import numpy as _np
        A = _np.vstack([_np.array(xs), _np.ones(len(xs))]).T
        m, _ = _np.linalg.lstsq(A, _np.array(ys), rcond=None)[0]
        a = float(_np.degrees(_np.arctan(m)))
        a = max(-4.0, min(4.0, a))
        return 0.0 if abs(a) < 0.15 else round(a, 2)
    return 0.0


_TIME_RE = _re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")


def _find_time_ref(img: Image.Image, words, box, W: int, H: int):
    """
    Hora vecina intacta a la derecha (ej. '20:45'): devuelve
    {cy, h} en pixeles globales (centro y alto de SU tinta) y su texto,
    para anclar tamano y linea base de la fecha nueva a ella.
    None si no hay hora confiable al lado.
    """
    cy = (box["y0"] + box["y1"]) / 2
    bh = max(1.0, box["y1"] - box["y0"])
    best, best_dx = None, 1e9
    for w in words:
        if w["conf"] < 15:
            continue
        if not _TIME_RE.match(w["text"].strip()):
            continue
        wcx = (w["l"] + w["w"] / 2) / W * 1000
        dx = wcx - box["x1"]
        if dx < -20 or dx > 260:
            continue
        wcy = (w["t"] + w["h"] / 2) / H * 1000
        if abs(wcy - cy) > max(bh * 0.6, 10):
            continue
        if dx < best_dx:
            best, best_dx = w, dx
    if best is None:
        return None
    try:
        crop = img.crop((best["l"], best["t"],
                         best["l"] + best["w"], best["t"] + best["h"])).convert("L")
        g = np.array(crop)
        _, bwi = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rows = np.where((bwi > 0).sum(axis=1) > max(1, bwi.shape[1] * 0.05))[0]
        if not len(rows):
            return None
        top, bot = int(rows[0]) + best["t"], int(rows[-1]) + best["t"]
        h = max(4, bot - top + 1)
        rgb = np.array(img.crop((best["l"], best["t"],
                                 best["l"] + best["w"],
                                 best["t"] + best["h"])).convert("RGB")).reshape(-1, 3)
        gg = rgb.mean(axis=1)
        dk = rgb[gg < np.percentile(gg, 5)]
        tink = tuple(int(v) for v in np.median(dk, axis=0)) if len(dk) else (60, 60, 60)
        return {"cy": (top + bot) / 2.0, "h": h, "text": best["text"], "ink": tink}
    except Exception:
        return None


def detect_registro_boxes(img: Image.Image, min_conf: float = 30.0):
    """Detecta fechas y filtra solo registro en cabecera. Un solo OCR.
    Devuelve ([{text, box, angle, ref}], omitidas). angle = renglon, ref = hora vecina."""
    words, W, H = _ocr_words(img)
    boxes = _build_boxes(words, W, H, min_conf)
    items = []
    for b in boxes:
        if _is_registro(b["box"], words, W, H):
            items.append({"text": b["text"], "box": b["box"],
                          "angle": _line_angle(words, b["box"], W, H),
                          "ref": _find_time_ref(img, words, b["box"], W, H)})
    return items, len(boxes) - len(items)


def _build_boxes(words, W: int, H: int, min_conf: float):

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


def detect_date_boxes(img: Image.Image, min_conf: float = 30.0):
    """
    Detecta TODAS las fechas visibles en la página (diagnostico; sin filtrar).
    Para reemplazo usar detect_registro_boxes().
    """
    words, W, H = _ocr_words(img)
    return _build_boxes(words, W, H, min_conf)


def _load_font(size: int, bold: bool = False):
    # Verdana = familia del formulario (NCC 0.79 vs 0.45 Arial). En Docker no existe
    # y cae al resto de la lista sin romper.
    regular = ["verdana.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
               "LiberationSerif-Regular.ttf"]
    bold_names = ["verdanab.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] + regular
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
        # grosor: erosion 3x3 depende del dpi (a 250 todo retiene mas).
        # Fechas/horas medidas: <=0.05 a 180dpi, <=0.135 a 250dpi -> regular.
        # Solo claramente pesado (>0.16) usa bold. Las fechas de estos docs son regular.
        eroded = cv2.erode(bin_inv, np.ones((3, 3), np.uint8), iterations=1)
        s_bin, s_ero = float(bin_inv.sum()), float(eroded.sum())
        ratio = (s_ero / s_bin) if s_bin > 0 else 0.0
        bold = ratio > 0.16
        # tinta: nucleo mas oscuro (p5) para igual contraste que la hora (core ~62-65)
        a = np.array(crop).reshape(-1, 3)
        gray = a.mean(axis=1)
        dark = a[gray < np.percentile(gray, 5)]
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
    """Grano del papel sobre el fondo ya limpio (ANTES del texto): sin blur,
    para no ablandar bordes. El texto se dibuja nitido encima, como impresion."""
    try:
        region = img.crop((x0, y0, x1, y1))
        arr = np.array(region).astype(np.int16)
        noise = np.random.normal(0, sigma * 0.5, arr.shape).astype(np.int16)
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
                        ink=None, bold: bool = False, target_h: int = 0,
                        skew_deg: float = 0.0, ref=None, stroke: int = 0):
    """
    Escribe `text` adaptado y nítido:
    - CON hora vecina (ref={cy,h} en px): tamano = altura de digito de la hora,
      centro de digitos alineado a su linea base. Fecha y hora quedan gemelas.
    - SIN hora: altura = text_h del original, centrado en el box.
    - ancho <=100% del box; supermuestreo 4x + LANCZOS; skew rota el texto.
    """
    from PIL import Image as _PI
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    if ink is None:
        ink = (25, 25, 25)
    SS = 4
    BW, BH = box_w * SS, box_h * SS
    if ref:
        hS = max(4 * SS, int(ref["h"] * SS))
        size_big = max(8 * SS, int(hS * 1.5))
        font_big = _load_font(size_big, bold=bold)
        d0 = ImageDraw.Draw(img)
        # Altura manda (gemela de la hora); el ancho se condensa, no se encoge.
        while size_big > 8 * SS:
            font_big = _load_font(size_big, bold=bold)
            try:
                bd = d0.textbbox((0, 0), text, font=font_big)
                b8 = d0.textbbox((0, 0), "8", font=font_big)
                dh = b8[3] - b8[1]
            except Exception:
                break
            if dh <= hS:
                break
            size_big -= max(1, SS // 2)
        try:
            bd = d0.textbbox((0, 0), text, font=font_big)
            b8 = d0.textbbox((0, 0), "8", font=font_big)
        except Exception:
            bd, b8 = (0, 0, BW // 2, BH // 2), (0, 0, BW // 4, BH // 4)
        tw_full = bd[2] - bd[0]
        sx = 1.0 if tw_full <= BW else max(0.80, BW / max(1, tw_full))
        tx = -bd[0]  # borde izquierdo = tinta original (conserva ritmo de espacios)
        dc_rel = (b8[1] + b8[3]) / 2.0 - bd[1]
        ty = (ref["cy"] * SS - y0 * SS) - dc_rel
        condense = sx
    else:
        want_h = target_h if target_h and target_h > 0 else int(box_h * 0.82)
        want_h = max(6, min(box_h, want_h))
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
        size_big = size * SS
        font_big = _load_font(size_big, bold=bold)
        d0 = ImageDraw.Draw(img)
        try:
            bd = d0.textbbox((0, 0), text, font=font_big)
            twb, thb = bd[2] - bd[0], bd[3] - bd[1]
        except Exception:
            twb, thb = BW // 2, BH // 2
            bd = (0, 0, twb, thb)
        tx = -bd[0]  # borde izquierdo = tinta original
        ty = (BH - thb) // 2 - bd[1]
        condense = 1.0
    # recorte ya con inpaint -> ampliar, dibujar grande, reducir (bordes suaves)
    try:
        crop_big = img.crop((x0, y0, x1, y1)).resize((BW, BH), _PI.BICUBIC)
    except Exception:
        crop_big = _PI.new("RGB", (BW, BH), (255, 255, 255))
    # texto en capa aparte para poder rotarlo segun el renglon sin mover el fondo
    txt_layer = _PI.new("RGBA", (BW, BH), (0, 0, 0, 0))
    d_big = ImageDraw.Draw(txt_layer)
    d_big.text((tx, ty), text, fill=ink + (255,), font=font_big,
               stroke_width=stroke, stroke_fill=ink + (255,))
    if condense < 1.0:
        # condensa horizontal al ancho del box: conserva altura gemela a la hora
        nw = max(1, int(BW * condense))
        txt_layer = txt_layer.resize((nw, BH), _PI.BICUBIC)
        canvas = _PI.new("RGBA", (BW, BH), (0, 0, 0, 0))
        canvas.alpha_composite(txt_layer, (0, 0))
        txt_layer = canvas
    if skew_deg:
        txt_layer = txt_layer.rotate(-skew_deg, resample=_PI.BICUBIC)
    crop_big = crop_big.convert("RGBA")
    crop_big.alpha_composite(txt_layer)
    crop_big = crop_big.convert("RGB")
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
    img = _scan_match(img, x0, y0, x1, y1, ana["bg_sigma"])
    img = _draw_fitted_text(img, x0, y0, x1, y1, text, ink=ana["ink"],
                            bold=ana["bold"], target_h=ana["text_h"])

    # --- pagina con su tamano original (no agrandar a pixeles) ---
    import img2pdf
    w0, h0 = pdfium.PdfDocument(pdf_bytes)[0].get_size()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pdf_out = img2pdf.convert(buf.getvalue(),
                              layout_fun=img2pdf.get_layout_fun((w0, h0)))
    return pdf_out


def replace_dates_in_pdf(pdf_bytes: bytes, new_text: str, dpi: int = 200):
    """
    Solo fechas de REGISTRO en cabecera: las elimina con inpaint y sobrepone
    `new_text` in-situ. Nacimiento/cuerpo quedan intactos.
    Devuelve (pdf_out, {total, skipped, pages, per_page})
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
    skipped_all = 0
    for i in range(n):
        img = render_page(pdf_bytes, i, dpi=dpi)
        W, H = img.size
        try:
            items, skipped = detect_registro_boxes(img)
        except Exception as e:
            raise RuntimeError(f"ocr no disponible (instala tesseract): {e}")
        skipped_all += skipped

        for b in items:
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
            img = _scan_match(img, x0, y0, x1, y1, ana["bg_sigma"])
            ref = b.get("ref")
            ink = (ref.get("ink") if ref and ref.get("ink") else ana["ink"])
            img = _draw_fitted_text(img, x0, y0, x1, y1, new_text, ink=ink,
                                    bold=ana["bold"], target_h=ana["text_h"],
                                    skew_deg=b.get("angle", 0.0), ref=ref,
                                    stroke=(1 if ref else 0))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_list.append(buf.getvalue())
        total += len(items)
        per_page.append({"page_index": i, "count": len(items),
                         "boxes": [{"text": b["text"], "box": b["box"],
                                    "angle": b.get("angle", 0.0),
                                    "ref_time": (b.get("ref") or {}).get("text")} for b in items]})

    import img2pdf
    w0, h0 = pdf[0].get_size()
    pdf_out = img2pdf.convert(*png_list, layout_fun=img2pdf.get_layout_fun((w0, h0)))
    return pdf_out, {"total": total, "skipped": skipped_all, "pages": n, "per_page": per_page}

def images_to_pdf(png_bytes_list: list[bytes], pagesize=(612.0, 792.0)) -> bytes:
    import img2pdf
    return img2pdf.convert(*png_bytes_list,
                           layout_fun=img2pdf.get_layout_fun(pagesize))
