import io
import os
import time
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

from .schemas import Box, OcrResponse
from .pdf_engine import render_page, apply_edit
from .db import log_job

load_dotenv()

FRONT_URL = os.getenv("FRONT_URL", "http://localhost:5173")

app = FastAPI(title="plagiobknd - motor edición documentos", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONT_URL, "http://localhost:5173", "https://*.vercel.app"],
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
    MVP con pytesseract. Prod: cambiar a PaddleOCR GPU.
    Devuelve boxes candidatas con regex de fecha.
    """
    import re
    data = await file.read()
    try:
        img = render_page(data, page, dpi=300)
    except Exception as e:
        raise HTTPException(400, f"render error: {e}")
    try:
        import pytesseract
        from pytesseract import Output
        d = pytesseract.image_to_data(img, lang="spa+eng", output_type=Output.DICT)
    except Exception as e:
        raise HTTPException(500, f"ocr no disponible (instala tesseract): {e}")

    W, H = img.size
    pattern = re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}")
    boxes = []
    n = len(d["text"])
    for i in range(n):
        t = (d["text"][i] or "").strip()
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if conf < 50 or not t:
            continue
        if pattern.search(t):
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            boxes.append({
                "text": t,
                "box": {
                    "x0": x / W * 1000, "y0": y / H * 1000,
                    "x1": (x + w) / W * 1000, "y1": (y + h) / H * 1000,
                },
                "conf": conf / 100.0,
            })
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
