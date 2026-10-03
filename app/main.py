import io
import os
import time
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

from .schemas import Box, OcrResponse
from .pdf_engine import render_page, apply_edit, detect_date_boxes, replace_dates_in_pdf
from .db import log_job

load_dotenv()

FRONT_URL = os.getenv("FRONT_URL", "http://localhost:5173")

app = FastAPI(title="plagiobknd - motor edición documentos", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONT_URL, "http://localhost:5173", "http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "service": "plagiobknd", "dpi": os.getenv("DPI_RENDER", "300")}

@app.post("/preview")
async def preview(file: UploadFile = File(...), page: int = Form(0), dpi: int = Form(150)):
    """Devuelve JPG rápido 150 DPI para el visor Konva."""
    data = await file.read()
    try:
        img = render_page(data, page, dpi=min(dpi, 200))
    except Exception as e:
        raise HTTPException(400, f"preview error: {e}")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

@app.post("/ocr-detect")
async def ocr_detect(file: UploadFile = File(...), page: int = Form(0)):
    """
    Detecta TODAS las fechas visibles (numéricas, ISO, abreviadas y '12 de marzo de 2026').
    Usa detect_date_boxes() compartido con /replace-dates.
    """
    data = await file.read()
    try:
        img = render_page(data, page, dpi=300)
    except Exception as e:
        raise HTTPException(400, f"render error: {e}")
    try:
        boxes = detect_date_boxes(img)
    except Exception as e:
        raise HTTPException(500, f"ocr no disponible (instala tesseract): {e}")

    return {"page_index": page, "boxes": boxes}

@app.post("/apply-edit")
async def apply_edit_endpoint(
    file: UploadFile = File(...),
    page_index: int = Form(0),
    x0: float = Form(...), y0: float = Form(...),
    x1: float = Form(...), y1: float = Form(...),
    text: str = Form(...),
):
    """Borrado pro con inpaint + overlay. Entrada en coords 0-1000 desde Konva."""
    data = await file.read()
    box = Box(x0=x0, y0=y0, x1=x1, y1=y1)
    try:
        out_pdf = apply_edit(data, page_index, box, text)
    except Exception as e:
        raise HTTPException(400, f"apply-edit error: {e}")

    log_job({
        "pdf_name": file.filename,
        "page_index": page_index,
        "new_text": text,
        "box": box.model_dump(),
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    return StreamingResponse(io.BytesIO(out_pdf), media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename=editado_p{page_index}.pdf"})


@app.post("/replace-dates")
async def replace_dates(
    file: UploadFile = File(...),
    new_text: str = Form(...),
    dpi: int = Form(200),
):
    """
    Reemplazo automático: solo fechas de REGISTRO en cabecera.
    Detecta fechas, filtra por etiqueta (FECHA/REGISTRO, sin NAC) en banda superior,
    las elimina con inpaint y sobrepone `new_text` in-situ. Nacimiento y cuerpo intactos.
    Devuelve el PDF COMPLETO.
    dpi 150=rápido, 200=equilibrado, 300=preciso (más lento).
    """
    t0 = time.time()
    new_text = (new_text or "").strip()
    if not new_text:
        raise HTTPException(400, "new_text vacío")
    if len(new_text) > 60:
        raise HTTPException(400, "new_text demasiado largo (máx 60)")
    dpi = max(100, min(300, int(dpi or 200)))
    data = await file.read()
    if not data:
        raise HTTPException(400, "PDF vacío")
    try:
        out_pdf, report = replace_dates_in_pdf(data, new_text, dpi=dpi)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"replace-dates error: {e}")

    log_job({
        "pdf_name": file.filename,
        "new_text": new_text,
        "total": report["total"],
        "skipped": report.get("skipped", 0),
        "pages": report["pages"],
        "per_page_counts": [p["count"] for p in report["per_page"]],
        "elapsed_s": round(time.time() - t0, 2),
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    base = (file.filename or "documento.pdf").rsplit(".", 1)[0]
    return StreamingResponse(
        io.BytesIO(out_pdf), media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={base}_fechas_{new_text.replace('/', '-')}.pdf",
            "X-Replacements-Total": str(report["total"]),
            "X-Pages": str(report["pages"]),
        },
    )
