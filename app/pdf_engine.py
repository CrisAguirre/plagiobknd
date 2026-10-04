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
from .profiles import get_profile


def _P(P):
    return P if P is not None else get_profile()

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


def _preprocess_for_ocr(img: Image.Image, P=None) -> Image.Image:
    """
    Escaneados con fondo gris: binariza con umbral adaptativo por página
    (media - k*std, clamped). Sin resize: mismas coords.
    """
    P = _P(P)
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    m, s = float(gray.mean()), float(gray.std())
    thv = int(m - P["pre_k"] * s)
    thv = max(P["pre_lo"], min(P["pre_hi"], thv))
    _, binimg = cv2.threshold(gray, thv, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binimg)


def _ocr_words(img: Image.Image, P=None):
    """OCR una vez por pagina: devuelve (words[{text,conf,l,t,w,h} px], W, H)."""
    P = _P(P)
    import pytesseract
    from pytesseract import Output
    _ensure_tesseract()
    ocr_img = _preprocess_for_ocr(img, P)
    d = pytesseract.image_to_data(ocr_img, lang=P["ocr_lang"], config=P["ocr_config"],
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


def _is_registro(box, words, W: int, H: int, P=None) -> bool:
    """
    Solo fechas de REGISTRO en cabecera:
    - banda superior (centro-y < top_band)
    - en su misma linea hay etiqueta kw_ok a la izquierda
    - si hay kw_no (FEC.NAC / nacimiento) -> NO, aunque diga fecha cerca
    - cuerpo del documento (accidente, evolucion, etc.) -> NO
    """
    P = _P(P)
    cy = (box["y0"] + box["y1"]) / 2
    if cy > P["top_band"]:
        return False
    bh = max(1.0, box["y1"] - box["y0"])
    # Decision POR RENGLON completo (sin ventana x): en una misma cabecera
    # conviven 'Fecha Solicitud', 'F. Realizacion' y 'F. Resultado'; con ventana
    # angosta el ultimo quedaba fuera y el reemplazo salia inconsistente.
    # La banda vertical angosta + top_band + exclusion NAC evitan falsos.
    line = []
    for w in words:
        if w["conf"] < P["word_conf"]:
            continue
        wcy = (w["t"] + w["h"] / 2) / H * 1000
        if abs(wcy - cy) > max(bh * P["ctx_band_k"], P["ctx_band_min"]):
            continue
        line.append(w["text"])
    t = " ".join(line).upper()
    if any(k in t for k in P["kw_no"]):
        return False
    return any(k in t for k in P["kw_ok"])


def _line_angle(words, box, W: int, H: int, P=None) -> float:
    """
    Inclinacion real del renglon donde vive la fecha (grados, + = baja a la derecha).
    Regresion sobre centros de palabras vecinas en banda angosta; por renglon,
    no por pagina (el escaneo/papel puede pandear distinto en cada zona).
    """
    P = _P(P)
    cy = (box["y0"] + box["y1"]) / 2
    bh = max(1.0, box["y1"] - box["y0"])
    xs, ys = [], []
    for w in words:
        if w["conf"] < P["word_conf"]:
            continue
        wcy = (w["t"] + w["h"] / 2) / H * 1000
        if abs(wcy - cy) > max(bh * P["ang_band_k"], P["ang_band_min"]):
            continue
        wcx = (w["l"] + w["w"] / 2) / W * 1000
        if box["x0"] - P["ctx_lo"] <= wcx <= box["x1"] + P["ctx_hi"]:
            xs.append(wcx); ys.append(wcy)
    if len(xs) >= 2 and (max(xs) - min(xs)) > P["ang_span"]:
        import numpy as _np
        A = _np.vstack([_np.array(xs), _np.ones(len(xs))]).T
        m, _ = _np.linalg.lstsq(A, _np.array(ys), rcond=None)[0]
        a = float(_np.degrees(_np.arctan(m)))
        a = max(-P["ang_clamp"], min(P["ang_clamp"], a))
        return 0.0 if abs(a) < P["ang_dead"] else round(a, 2)
    return 0.0


_TIME_RE = _re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")


def _find_time_ref(img: Image.Image, words, box, W: int, H: int, P=None):
    """
    Hora vecina intacta a la derecha (ej. '20:45'): devuelve
    {cy, h} en pixeles globales (centro y alto de SU tinta) y su texto,
    para anclar tamano y linea base de la fecha nueva a ella.
    None si no hay hora confiable al lado.
    """
    P = _P(P)
    cy = (box["y0"] + box["y1"]) / 2
    bh = max(1.0, box["y1"] - box["y0"])
    best, best_dx = None, 1e9
    for w in words:
        if w["conf"] < P["word_conf"]:
            continue
        if not _TIME_RE.match(w["text"].strip()):
            continue
        wcx = (w["l"] + w["w"] / 2) / W * 1000
        dx = wcx - box["x1"]
        if dx < P["time_dx"][0] or dx > P["time_dx"][1]:
            continue
        wcy = (w["t"] + w["h"] / 2) / H * 1000
        if abs(wcy - cy) > max(bh * P["time_band_k"], P["time_band_min"]):
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
        rows = np.where((bwi > 0).sum(axis=1) > max(1, bwi.shape[1] * P["row_frac"]))[0]
        if not len(rows):
            return None
        top, bot = int(rows[0]) + best["t"], int(rows[-1]) + best["t"]
        h = max(4, bot - top + 1)
        rgb = np.array(img.crop((best["l"], best["t"],
                                 best["l"] + best["w"],
                                 best["t"] + best["h"])).convert("RGB")).reshape(-1, 3)
        gg = rgb.mean(axis=1)
        dk = rgb[gg < np.percentile(gg, P["ink_pct"])]
        tink = tuple(int(v) for v in np.median(dk, axis=0)) if len(dk) else P["tink_fallback"]
        return {"cy": (top + bot) / 2.0, "h": h, "text": best["text"], "ink": tink,
                "l": best["l"]}
    except Exception:
        return None


def _tighten_box_y(img: Image.Image, x0: int, y0: int, x1: int, y1: int):
    """
    Caja OCR alta que abarca varios renglones (p8): recorta y al grupo de tinta
    que contiene el centro vertical. Solo encoge, nunca expande. Si no hay grupo
    claro, devuelve la caja intacta.
    """
    try:
        crop = img.crop((x0, y0, x1, y1)).convert("L")
        g = np.array(crop)
        _, bwi = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        has = (bwi > 0).sum(axis=1) > max(1, bwi.shape[1] * 0.04)
        if not has.any():
            return y0, y1
        # grupos contiguos (corte con >3px vacios)
        groups, s = [], None
        for r, v in enumerate(has):
            if v and s is None:
                s = r
            if not v and s is not None:
                groups.append((s, r - 1))
                s = None
        if s is not None:
            groups.append((s, len(has) - 1))
        cy = (y1 - y0) / 2.0
        best = None
        for g0, g1 in groups:
            if g0 - 1 <= cy <= g1 + 1 and (g1 - g0 + 1) >= 6:
                best = (g0, g1)
                break
        if best is None:
            return y0, y1
        return max(y0, y0 + best[0] - 1), min(y1, y0 + best[1] + 1)
    except Exception:
        return y0, y1


def _font_frac(datepart: str, full: str) -> float:
    """Fraccion del ancho que ocupa la fecha dentro del token pegado,
    medida con la misma fuente del dibujo (independiente del tamano)."""
    try:
        from PIL import ImageFont as _IF
        f = None
        for name in ("verdana.ttf", "arial.ttf", "DejaVuSans.ttf"):
            try:
                f = _IF.truetype(name, 100)
                break
            except Exception:
                continue
        if f is None:
            return len(datepart) / max(1, len(full))
        from PIL import ImageDraw as _ID, Image as _IM
        d = _ID.Draw(_IM.new("L", (10, 10)))
        wd = d.textlength(datepart, font=f)
        wf = d.textlength(full, font=f)
        return max(0.5, min(0.97, wd / max(1.0, wf)))
    except Exception:
        return len(datepart) / max(1, len(full))


def _extract_suffix(t: str):
    """
    Token pegado ('06/05/2023:08145', '23/03/2023,', '06/05/2023.'):
    devuelve el sufijo tras la fecha (para reporte). Los pixeles del sufijo
    se PRESERVAN (no se redibujan): el corte usa fraccion de fuente, no de
    caracteres (los digitos son mas anchos que ':'/'.'/',' y el corte por
    caracteres mutilaba el ultimo digito).
    Solo sufijos de puntuacion/hora; con letras se ignora.
    """
    for p in _SINGLE_PATS:
        m = p.match(t)
        if m and m.end() < len(t):
            suf = t[m.end():]
            if suf and len(suf) <= 10 and _re.fullmatch(r"[:;.,\s\d]+", suf):
                return suf
    return ""


def _right_gap(img: Image.Image, x1: int, y0: int, y1: int, cap: int):
    """
    Espacio libre a la derecha de la fecha (hasta la proxima tinta: hora,
    etiqueta o borde), con 2px de seguridad. Para extender el lienzo sin
    tocar vecinos. 0 si no hay.
    """
    try:
        W, _ = img.size
        x1c = max(0, min(W - 1, x1))
        x2c = max(x1c, min(W, x1 + max(0, cap)))
        if x2c <= x1c:
            return 0
        g = np.array(img.crop((x1c, y0, x2c, y1)).convert("L"))
        _, bwi = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cols = np.where((bwi > 0).sum(axis=0) > max(1, bwi.shape[0] * 0.1))[0]
        first = int(cols[0]) if len(cols) else (x2c - x1c)
        return max(0, min(cap, first - 2))
    except Exception:
        return 0
    """Box 0-1000 -> pixeles, recortando sufijo pegado con fraccion de fuente."""
    x0 = max(0, int(bx["x0"] / 1000.0 * W))
    y0 = max(0, int(bx["y0"] / 1000.0 * H))
    x1 = min(W, int(bx["x1"] / 1000.0 * W))
    y1 = min(H, int(bx["y1"] / 1000.0 * H))
    suf = _extract_suffix(text)
    if suf:
        for p in _SINGLE_PATS:
            m = p.match(text)
            if m and m.end() < len(text):
                frac = _font_frac(text[:m.end()], text)
                x1 = max(x0 + 1, int(x0 + (x1 - x0) * frac) - 1)
                break
    return x0, y0, x1, y1, suf


def detect_registro_boxes(img: Image.Image, min_conf=None, P=None):
    """Detecta fechas y filtra solo registro en cabecera. Un solo OCR.
    Devuelve ([{text, box, angle, ref}], omitidas). angle = renglon, ref = hora vecina."""
    P = _P(P)
    if min_conf is None:
        min_conf = P["min_conf"]
    words, W, H = _ocr_words(img, P)
    boxes = _build_boxes(words, W, H, min_conf)
    items = []
    for b in boxes:
        if _is_registro(b["box"], words, W, H, P):
            items.append({"text": b["text"], "box": b["box"],
                          "angle": _line_angle(words, b["box"], W, H, P),
                          "ref": _find_time_ref(img, words, b["box"], W, H, P)})
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
            # segunda oportunidad: fecha incrustada en token pegado con conf
            # baja ('Resuitados:06/05/2023:10'). Caja refinada a la fecha.
            if w["conf"] >= 15:
                for p in (P_NUM, P_ISO):
                    m = p.search(w["text"])
                    if m:
                        try:
                            from PIL import ImageFont as _IF2, ImageDraw as _ID2, Image as _IM2
                            f2 = None
                            for _fn in ("verdana.ttf", "arial.ttf", "DejaVuSans.ttf"):
                                try:
                                    f2 = _IF2.truetype(_fn, 100)
                                    break
                                except Exception:
                                    continue
                            if f2 is None:
                                break
                            _dd = _ID2.Draw(_IM2.new("L", (10, 10)))
                            t = w["text"]
                            f0 = _dd.textlength(t[:m.start()], font=f2)
                            f1 = _dd.textlength(t[:m.end()], font=f2)
                            ft = _dd.textlength(t, font=f2)
                            nx0 = w["l"] + w["w"] * f0 / max(1.0, ft)
                            nx1 = w["l"] + w["w"] * f1 / max(1.0, ft)
                            if nx1 - nx0 > 8:
                                found.append(to1000(nx0, w["t"], nx1 - nx0, w["h"],
                                                    w["conf"] * 0.85, t[m.start():m.end()]))
                        except Exception:
                            pass
                        break
            continue
        if _is_date_token(w["text"]):
            cand = to1000(w["l"], w["t"], w["w"], w["h"], w["conf"], w["text"])
            # dedup: tesseract a veces devuelve la misma palabra 2 veces
            # (doble dibujo = doble tinta = negrita fantasma)
            dup = False
            for f in found:
                a, o = f["box"], cand["box"]
                ix0, iy0 = max(a["x0"], o["x0"]), max(a["y0"], o["y0"])
                ix1, iy1 = min(a["x1"], o["x1"]), min(a["y1"], o["y1"])
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    area = min((a["x1"] - a["x0"]) * (a["y1"] - a["y0"]),
                               (o["x1"] - o["x0"]) * (o["y1"] - o["y0"]))
                    if area and inter / area > 0.5:
                        dup = True
                        break
            if not dup:
                found.append(cand)

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


def detect_date_boxes(img: Image.Image, min_conf=None, P=None):
    """
    Detecta TODAS las fechas visibles en la página (diagnostico; sin filtrar).
    Para reemplazo usar detect_registro_boxes().
    """
    P = _P(P)
    if min_conf is None:
        min_conf = P["min_conf"]
    words, W, H = _ocr_words(img, P)
    return _build_boxes(words, W, H, min_conf)


def _load_font(size: int, bold: bool = False, P=None):
    # Orden = prioridad (ver perfil base). Si falta en el sistema cae al siguiente.
    P = _P(P)
    regular = list(P["fonts_regular"])
    bold_names = list(P["fonts_bold"]) + regular
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


def _analyze_orig(img: Image.Image, x0: int, y0: int, x1: int, y1: int, P=None):
    """
    Mide el texto ORIGINAL antes de borrarlo:
    - text_h/top: filas reales de tinta (sin padding Tesseract)
    - bold: por grosor de trazo (erosion ratio), no por posicion
    - ink: color nucleo oscuro
    - bg_sigma: grano del fondo vecino (anillo) para igualar nitidez
    Fallbacks seguros si algo falla.
    """
    P = _P(P)
    box_w = max(1, x1 - x0); box_h = max(1, y1 - y0)
    try:
        crop = img.crop((max(0, x0), max(0, y0), x1, y1)).convert("RGB")
        g = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
        _, bin_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = bin_inv.shape
        # filas con tinta (ignora motas)
        rows = np.where((bin_inv > 0).sum(axis=1) > max(1, w * P["row_min_frac"]))[0]
        text_h = int(rows[-1] - rows[0] + 1) if len(rows) else int(box_h * 0.7)
        text_h = max(6, min(box_h, text_h))
        top = int(rows[0]) if len(rows) else 0
        # grosor por erosion (depende del dpi; ver perfil base calibrado 180+250)
        eroded = cv2.erode(bin_inv, np.ones((3, 3), np.uint8), iterations=1)
        s_bin, s_ero = float(bin_inv.sum()), float(eroded.sum())
        ratio = (s_ero / s_bin) if s_bin > 0 else 0.0
        bold = ratio > P["bold_thr"]
        # tinta: nucleo oscuro para igual contraste que la hora
        a = np.array(crop).reshape(-1, 3)
        gray = a.mean(axis=1)
        dark = a[gray < np.percentile(gray, P["ink_pct_fb"])]
        ink = tuple(int(v) for v in np.median(dark, axis=0)) if len(dark) else P["ink_fallback"]
        ink = tuple(max(0, min(255, int(v * P["ink_dark_k"]))) for v in ink)
    except Exception:
        text_h, bold, ink = int(box_h * 0.7), False, P["ink_fallback"]
        top = 0
    try:
        W, H = img.size
        r = P["ring"]
        rx0, ry0 = max(0, x0 - r), max(0, y0 - r)
        rx1, ry1 = min(W, x1 + r), min(H, y1 + r)
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
            bg_sigma = P["bg_fallback"]
        bg_sigma = max(P["bg_lo"], min(P["bg_hi"], bg_sigma))
    except Exception:
        bg_sigma = P["bg_fallback"]
    return {"text_h": text_h, "top": top, "bold": bold, "ink": ink, "bg_sigma": bg_sigma}


def _scan_match(img: Image.Image, x0: int, y0: int, x1: int, y1: int, sigma: float, P=None):
    """Grano del papel sobre el fondo ya limpio (ANTES del texto): sin blur,
    para no ablandar bordes. El texto se dibuja nitido encima, como impresion."""
    P = _P(P)
    try:
        region = img.crop((x0, y0, x1, y1))
        arr = np.array(region).astype(np.int16)
        noise = np.random.normal(0, sigma * P["grain_k"], arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(arr), (x0, y0))
    except Exception:
        pass
    return img


def _inpaint_box(img: Image.Image, x0: int, y0: int, x1: int, y1: int, P=None) -> Image.Image:
    """Elimina el contenido del box reconstruyendo el fondo (inpaint TELEA)."""
    P = _P(P)
    W, H = img.size
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    mask = np.zeros((H, W), dtype=np.uint8)
    pad = max(2, int((x1 - x0) * P["inpaint_pad"]))
    cv2.rectangle(mask, (max(0, x0 - pad), max(0, y0 - pad)),
                  (min(W, x1 + pad), min(H, y1 + pad)), 255, -1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(cv_img, mask, 4, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))


def _draw_fitted_text(img: Image.Image, x0: int, y0: int, x1: int, y1: int, text: str,
                        ink=None, bold: bool = False, target_h: int = 0,
                        skew_deg: float = 0.0, ref=None, stroke: int = 0,
                        clip=None, P=None):
    """
    Escribe `text` adaptado y nítido:
    - CON hora vecina (ref={cy,h} en px): tamano = altura de digito de la hora,
      base de digitos anclada a la base de la hora. Fecha y hora quedan gemelas.
    - SIN hora: altura = text_h del original, anclado a borde izquierdo.
    - clip=(top_rel,h): recorta tinta a las filas originales (las barras de Verdana
      son mas largas que las del formulario); sin esto desbordan 3px abajo.
    """
    from PIL import Image as _PI
    P = _P(P)
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    if ink is None:
        ink = P["ink_fallback"]
    SS = P["ss"]
    BW, BH = box_w * SS, box_h * SS
    if ref:
        hS = max(4 * SS, int(ref["h"] * SS))
        size_big = max(8 * SS, int(hS * P["size_k_ref"]))
        font_big = _load_font(size_big, bold=bold, P=P)
        d0 = ImageDraw.Draw(img)
        # Altura manda (gemela de la hora); el ancho se condensa, no se encoge.
        while size_big > 8 * SS:
            font_big = _load_font(size_big, bold=bold, P=P)
            try:
                bd = d0.textbbox((0, 0), text, font=font_big)
                b8 = d0.textbbox((0, 0), "8", font=font_big)
                dh = b8[3] - b8[1]
            except Exception:
                break
            if dh <= hS:
                break
            size_big -= max(1, SS // P["size_step_div"])
        try:
            bd = d0.textbbox((0, 0), text, font=font_big)
            b8 = d0.textbbox((0, 0), "8", font=font_big)
        except Exception:
            bd, b8 = (0, 0, BW // 2, BH // 2), (0, 0, BW // 4, BH // 4)
        tw_full = bd[2] - bd[0]
        sx = 1.0 if tw_full <= BW else max(P["condense_min"], BW / max(1, tw_full))
        tx = -bd[0]  # borde izquierdo = tinta original (tinta queda en x=0)
        # Base de digitos ("8") sobre la base de la hora. OJO: al dibujar en (tx,ty)
        # la tinta cae en ty+bb, asi que ty = objetivo - b8[3] (sin restar bd[1] dos veces).
        ty = (ref["cy"] * SS + (ref["h"] * SS) / 2.0 - y0 * SS) - b8[3]
        condense = sx
    else:
        want_h = target_h if target_h and target_h > 0 else int(box_h * P["target_fallback_k"])
        want_h = max(6, min(box_h, want_h))
        probe = ImageDraw.Draw(img)
        size = max(8, int(want_h * P["size_k"]))
        font = _load_font(size, bold=bold, P=P)
        tw = th = 0
        while size > 8:
            font = _load_font(size, bold=bold, P=P)
            try:
                bb = probe.textbbox((0, 0), text, font=font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
            except Exception:
                try:
                    tw = int(font.getlength(text)); th = size
                except Exception:
                    break
            if tw <= box_w * P["width_cap"] and th <= want_h and th >= want_h - 1:
                break
            if tw <= box_w * P["width_cap"] and th <= want_h:
                break
            size -= 1
        size_big = size * SS
        font_big = _load_font(size_big, bold=bold, P=P)
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
    # ablanda solo bordes del texto al nivel del escaneo (radio a escala SS;
    # text_blur esta calibrado a SS=4). No toca el fondo: ese ya tiene grano.
    try:
        from PIL import ImageFilter as _IF2
        txt_layer = txt_layer.filter(_IF2.GaussianBlur(P["text_blur"] * SS / 4.0))
    except Exception:
        pass
    if clip:
        # tinta solo donde la habia: corta colas de barras fuera de filas originales
        ct, chh = max(0, int(clip[0] * SS)), int(clip[1] * SS)
        alpha = txt_layer.getchannel("A")
        blk = _PI.new("L", alpha.size, 0)
        blk.paste(alpha.crop((0, ct, BW, min(BH, ct + chh))), (0, ct))
        txt_layer.putalpha(blk)
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

def apply_edit(pdf_bytes: bytes, page_index: int, box, text: str, profile=None) -> bytes:
    """
    1. render DPI del perfil
    2. inpaint zona (borra fecha vieja reconstruyendo fondo)
    3. overlay texto nuevo auto-ajustado y centrado
    4. re-empaqueta solo esa página a PDF (mismo tamaño de página original)
    Devuelve PDF de 1 página. Para documento completo usar replace_dates_in_pdf().
    """
    P = profile if isinstance(profile, dict) else get_profile(profile)
    img = render_page(pdf_bytes, page_index, dpi=P["dpi"])
    W, H = img.size
    x0, y0, x1, y1 = _box_to_pixels(box, W, H)

    ana = _analyze_orig(img, x0, y0, x1, y1, P)
    img = _inpaint_box(img, x0, y0, x1, y1, P)
    img = _scan_match(img, x0, y0, x1, y1, ana["bg_sigma"], P)
    img = _draw_fitted_text(img, x0, y0, x1, y1, text, ink=ana["ink"],
                            bold=ana["bold"], target_h=ana["text_h"],
                            clip=(ana["top"], ana["text_h"]), P=P)

    # --- pagina con su tamano original (no agrandar a pixeles) ---
    import img2pdf
    w0, h0 = pdfium.PdfDocument(pdf_bytes)[0].get_size()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pdf_out = img2pdf.convert(buf.getvalue(),
                              layout_fun=img2pdf.get_layout_fun((w0, h0)))
    return pdf_out


def replace_dates_in_pdf(pdf_bytes: bytes, new_text: str, dpi=None, profile=None):
    """
    Solo fechas de REGISTRO en cabecera: las elimina con inpaint y sobrepone
    `new_text` in-situ. Nacimiento/cuerpo quedan intactos.
    profile: nombre en profiles.py ("base" = modelo congelado del doc 1) o dict.
    Devuelve (pdf_out, {total, skipped, pages, per_page})
    """
    P = profile if isinstance(profile, dict) else get_profile(profile)
    new_text = (new_text or "").strip()
    if not new_text:
        raise ValueError("new_text vacío")
    dpi = max(100, min(300, int(dpi or P["dpi"])))

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
            items, skipped = detect_registro_boxes(img, P=P)
        except Exception as e:
            raise RuntimeError(f"ocr no disponible (instala tesseract): {e}")
        skipped_all += skipped

        for b in items:
            x0, y0, x1, y1, suf = _box_pixels(b["box"], b["text"], W, H)
            if x1 <= x0 or y1 <= y0:
                continue
            y0, y1 = _tighten_box_y(img, x0, y0, x1, y1)
            if y1 <= y0:
                continue
            ana = _analyze_orig(img, x0, y0, x1, y1, P)
            img = _inpaint_box(img, x0, y0, x1, y1, P)
            img = _scan_match(img, x0, y0, x1, y1, ana["bg_sigma"], P)
            ref = b.get("ref")
            ink = (ref.get("ink") if ref and ref.get("ink") else ana["ink"])
            img = _draw_fitted_text(img, x0, y0, x1, y1, new_text, ink=ink,
                                    bold=ana["bold"], target_h=ana["text_h"],
                                    skew_deg=b.get("angle", 0.0), ref=ref,
                                    stroke=(P["stroke_ref"] if ref else 0),
                                    clip=(ana["top"], ana["text_h"]), P=P)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_list.append(buf.getvalue())
        total += len(items)
        per_page.append({"page_index": i, "count": len(items),
                         "boxes": [{"text": b["text"], "box": b["box"],
                                    "angle": b.get("angle", 0.0),
                                    "ref_time": (b.get("ref") or {}).get("text"),
                                    "suffix": _extract_suffix(b["text"])} for b in items]})

    import img2pdf
    w0, h0 = pdf[0].get_size()
    pdf_out = img2pdf.convert(*png_list, layout_fun=img2pdf.get_layout_fun((w0, h0)))
    return pdf_out, {"total": total, "skipped": skipped_all, "pages": n, "per_page": per_page}

def images_to_pdf(png_bytes_list: list[bytes], pagesize=(612.0, 792.0)) -> bytes:
    import img2pdf
    return img2pdf.convert(*png_bytes_list,
                           layout_fun=img2pdf.get_layout_fun(pagesize))
