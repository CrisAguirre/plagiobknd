"""
Perfiles de edicion por documento.

BASE = modelo congelado que produjo el doc 1 (v13, aprobado).
NO MODIFICAR: cualquier ajuste para docs 2-4 va en un perfil nuevo
(DOC2, DOC3, DOC4) que hereda de BASE y solo cambia lo medido.
"""

BASE = dict(
    # render: escaneos nativos ~200dpi -> 1:1 sin remuestreo
    dpi=200,
    # tipografias (orden = prioridad; en Docker cae al resto sin romper)
    fonts_regular=[
        "verdana.ttf", "arial.ttf", "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf", "LiberationSerif-Regular.ttf",
    ],
    fonts_bold=[
        "verdanab.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf",
    ],
    # OCR
    ocr_lang="spa+eng",
    ocr_config="--oem 1 --psm 6",
    min_conf=30.0,
    word_conf=15.0,
    # binarizacion previa al OCR: umbral = media - k*std, clamped
    pre_k=0.8,
    pre_lo=140,
    pre_hi=200,
    # filtro registro: banda superior (centro-y < top_band, coords 0-1000)
    top_band=400.0,
    # ventana de contexto en la linea: [x0-ctx_lo, x1+ctx_hi]
    ctx_lo=380.0,
    ctx_hi=150.0,
    # banda vertical contexto: max(bh*ctx_band_k, ctx_band_min)
    ctx_band_k=0.9,
    ctx_band_min=14.0,
    kw_no=("NAC",),
    kw_ok=("REGISTRO", "FECHA"),
    # banda vertical para angulo: max(bh*ang_band_k, ang_band_min)
    ang_band_k=0.5,
    ang_band_min=8.0,
    ang_span=120.0,
    ang_clamp=4.0,
    ang_dead=0.15,
    # hora vecina: dx en [time_dx], banda max(bh*time_band_k, time_band_min)
    time_dx=(-20.0, 260.0),
    time_band_k=0.6,
    time_band_min=10.0,
    # filas de tinta: fraccion minima de Nein (ref) / analisis (row_min_frac)
    row_frac=0.05,
    row_min_frac=0.04,
    # tinta analisis: percentil oscuro x0.85 (fallback sin hora necesita nucleo real)
    ink_pct_fb=3,
    ink_dark_k=0.85,
    # negrita por erosion 3x3 (depende del dpi; calibrado 180+250)
    bold_thr=0.16,
    # tinta: percentil oscuro + fallbacks
    ink_pct=5,
    ink_fallback=(25, 25, 25),
    tink_fallback=(60, 60, 60),
    # anillo de papel para grano
    ring=12,
    bg_lo=1.2,
    bg_hi=5.0,
    bg_fallback=2.5,
    # inpaint: pad relativo al ancho + dilate 1
    inpaint_pad=0.03,
    # dibujo
    ss=4,
    size_k=1.45,
    size_k_ref=1.5,
    size_step_div=2,  # decremento = max(1, SS//size_step_div) en ref
    width_cap=1.0,
    condense_min=0.80,
    stroke_ref=1,
    text_blur=1.6,
    grain_k=0.5,
    target_fallback_k=0.82,
)

# Activo por defecto. Perfiles DOC2/3/4 se agregan aqui tras medir cada doc.
PROFILES = {
    "base": BASE,
}

_ACTIVE = "base"


def use_profile(name: str):
    global _ACTIVE
    if name not in PROFILES:
        raise ValueError(f"perfil desconocido: {name}")
    _ACTIVE = name


def get_profile(name=None):
    return PROFILES[name or _ACTIVE]


def active_name():
    return _ACTIVE
