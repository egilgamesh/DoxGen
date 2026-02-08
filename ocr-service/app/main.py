import io, fitz
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from PIL import Image
from paddleocr import PaddleOCR

app = FastAPI(title="OCR Service")
ocr = PaddleOCR(use_angle_cls=True, lang="en")

class OcrRequest(BaseModel):
    tenantId: str
    documentId: str
    fileUrl: str

@app.post("/ocr/extract")
def extract(req: OcrRequest):
    r = httpx.get(req.fileUrl, timeout=120)
    r.raise_for_status()

    doc = fitz.open(stream=r.content, filetype="pdf")
    pages = []

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()

        if not text:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            res = ocr.ocr(img, cls=True)
            lines = []
            for block in res or []:
                for item in block:
                    lines.append(item[1][0])
            text = "\n".join(lines).strip()

        pages.append({"page": i + 1, "text": text})

    return {"tenantId": req.tenantId, "documentId": req.documentId, "pages": pages}
