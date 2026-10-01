# plagiobknd — backend

Motor profesional FastAPI para edición de PDFs escaneados. Despliegue en Render + Mongo Atlas.

## Endpoints
- `GET /health`
- `POST /preview` (file, page, dpi) → JPG rápido para Konva
- `POST /ocr-detect` (file, page) → boxes fecha candidatas
- `POST /apply-edit` (file, page_index, x0,y0,x1,y1 0-1000, text) → PDF 1 pág con inpaint+overlay

## Dev
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Necesita Tesseract instalado para `/ocr-detect` (en Docker ya viene).

## Deploy Render
- Runtime: Docker (`Dockerfile`)
- Env: `MONGO_URI`, `MONGO_DB=plagio`, `FRONT_URL=https://tu-front.vercel.app`, `DPI_RENDER=300`
- Front en Vercel apunta con `VITE_API_URL=https://tu-backend.onrender.com`

## Notas
- Render 300 DPI, `pypdfium2` evita AGPL de PyMuPDF.
- Originales nunca se sobrescriben, se audita en Mongo `jobs`.
