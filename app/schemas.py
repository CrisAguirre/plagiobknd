from pydantic import BaseModel
from typing import List, Optional

class Box(BaseModel):
    """Coordenadas normalizadas 0-1000 como las envía Konva en front."""
    x0: float
    y0: float
    x1: float
    y1: float

class EditRequest(BaseModel):
    page_index: int = 0
    box: Box
    text: str
    font_size_ratio: float = 0.8  # respecto a alto de box
    # futuro: font_hint, color_hint, lang

class OcrBox(BaseModel):
    text: str
    box: Box
    conf: float

class OcrResponse(BaseModel):
    page_index: int
    boxes: List[OcrBox]

class JobLog(BaseModel):
    pdf_name: str
    page_index: int
    old_text: Optional[str] = None
    new_text: str
