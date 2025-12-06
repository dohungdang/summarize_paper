from pathlib import Path
from typing import Union
import fitz
from fastapi import UploadFile

def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Đọc toàn bộ text từ file PDF """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()

async def extract_text_from_upload(file: UploadFile) -> str:
    """ Helper cho FastAPI: nhận UploadFile (PDF), lưu tạm rồi đọc text """
    tmp_dir = Path(__file__).resolve().parents[2] / "tmp_pdfs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / (file.filename or "upload.pdf")
    contents = await file.read()
    tmp_path.write_bytes(contents)
    try:
        text = extract_text_from_pdf(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return text
