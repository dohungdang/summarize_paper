from datetime import datetime
from typing import List, Literal, Optional
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from uuid import uuid4
from database import Base, engine, get_db
from models import Summary
from rag import chunker, loader, retriever, summarizer, vector_store

# Khởi tạo bảng
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Paper Summarizer API",
    version="0.2.0",
)

# CORS cho dev (frontend chạy localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SCHEMA

class SummarizeRequest(BaseModel):
    text: str
    length: Literal["short", "medium", "long"] = "medium"

class SummaryResponse(BaseModel):
    id: int
    summary: str

    class Config:
        orm_mode = True

class SummaryListItem(BaseModel):
    id: int
    source_type: str
    paper_id: Optional[str] = None
    length: str
    created_at: datetime
    preview: str

    class Config:
        orm_mode = True

class SummaryStats(BaseModel):
    total_summaries: int
    total_text: int
    total_pdf: int

class UploadPaperResponse(BaseModel):
    paper_id: str
    num_chunks: int
    summary_id: int
    summary: str

# BASIC ROUTES

@app.get("/")
def root():
    return {"status": "ok", "message": "Summarizer backend is running"}

# TEXT SUMMARIZATION 

@app.post("/summarize_text", response_model=SummaryResponse)
def summarize_text_endpoint(
    payload: SummarizeRequest,
    db: Session = Depends(get_db),
):
    """ Tóm tắt đoạn text ngắn, không dùng RAG """
    try:
        summary_text = summarizer.summarize_text(payload.text, payload.length)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}")

    if not summary_text:
        raise HTTPException(status_code=500, detail="Model returned empty summary")

    db_obj = Summary(
        source_type="text",
        paper_id=None,
        input_text=payload.text,
        summary=summary_text,
        length=payload.length,
        model_name=summarizer.MODEL_NAME,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)

    return SummaryResponse(id=db_obj.id, summary=db_obj.summary)

# RAG + PDF UPLOAD 

@app.post("/upload_paper", response_model=UploadPaperResponse)
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """ Upload 1 file PDF: Extract text, chunk + index vào Chroma, dùng RAG tóm tắt nội dung chính, lưu lịch sử vào SQLite """
    if file.content_type not in (
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Extract text
    text = await loader.extract_text_from_upload(file)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty or unreadable PDF")

    # Chunk text
    chunks = chunker.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Failed to create chunks from PDF")

    # Index vào Chroma
    paper_id = str(uuid4())
    num_chunks = vector_store.add_chunks(paper_id, chunks)

    # Dùng RAG tóm tắt nhanh
    generic_question = "Tóm tắt ngắn gọn nội dung chính của tài liệu."
    top_chunks = retriever.get_relevant_chunks(
        paper_id=paper_id,
        query=generic_question,
        k=min(5, len(chunks)),
    )
    context = "\n\n".join(top_chunks) if top_chunks else text

    summary_text = summarizer.summarize_text(context, length="medium")

    # Lưu DB 
    preview_input = text[:4000]

    db_obj = Summary(
        source_type="pdf",
        paper_id=paper_id,
        input_text=preview_input,
        summary=summary_text,
        length="medium",
        model_name=summarizer.MODEL_NAME,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)

    return UploadPaperResponse(
        paper_id=paper_id,
        num_chunks=num_chunks,
        summary_id=db_obj.id,
        summary=db_obj.summary,
    )

# HISTORY và STATS 

@app.get("/summaries", response_model=List[SummaryListItem])
def list_summaries(
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """ Lấy danh sách lịch sử tóm tắt (dạng list) """
    query = (
        db.query(Summary)
        .order_by(Summary.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    items: List[SummaryListItem] = []
    for row in query.all():
        preview = (row.input_text or "")[:200]
        items.append(
            SummaryListItem(
                id=row.id,
                source_type=row.source_type,
                paper_id=row.paper_id,
                length=row.length,
                created_at=row.created_at,
                preview=preview,
            )
        )

    return items


@app.get("/summaries/{summary_id}", response_model=SummaryResponse)
def get_summary(
    summary_id: int,
    db: Session = Depends(get_db),
):
    """ Lấy chi tiết 1 bản tóm tắt theo id """
    obj = db.query(Summary).filter(Summary.id == summary_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Summary not found")

    return SummaryResponse(id=obj.id, summary=obj.summary)


@app.get("/summaries/stats", response_model=SummaryStats)
def get_summary_stats(
    db: Session = Depends(get_db),
):
    """" API thống kê đơn giản:tổng số lần tóm tắt, số lần tóm tắt text thường, số lần tóm tắt PDF """
    total = db.query(func.count(Summary.id)).scalar() or 0
    total_text = (
        db.query(func.count(Summary.id))
        .filter(Summary.source_type == "text")
        .scalar()
        or 0
    )
    total_pdf = (
        db.query(func.count(Summary.id))
        .filter(Summary.source_type == "pdf")
        .scalar()
        or 0
    )

    return SummaryStats(
        total_summaries=total,
        total_text=total_text,
        total_pdf=total_pdf,
    )
