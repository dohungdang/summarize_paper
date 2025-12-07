import os
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

# Tắt telemetry ngay từ đầu (phòng trường hợp Chroma vẫn đọc env)
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"

# Patch thẳng telemetry.capture để không còn gọi PostHog nữa
try:
    import chromadb.telemetry as _chroma_telemetry

    def _noop_capture(*args, **kwargs):
        return None

    _chroma_telemetry.capture = _noop_capture
except Exception:
    pass

DB_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(
    path=str(DB_DIR),
)

collection = client.get_or_create_collection(
    name="paper_chunks",
    metadata={"hnsw:space": "cosine"},
)

def add_chunks(paper_id: str, chunks: list) -> int:
    """ Thêm các đoạn văn bản (chunks) vào ChromaDB """
    # Chuẩn bị các chunk cho việc lưu trữ
    ids = [f"{paper_id}_{i}" for i in range(len(chunks))]
    documents = chunks 
    metadatas = [{"source": paper_id} for _ in range(len(chunks))]
    # Lưu các chunk vào collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)  # Trả về số lượng chunk đã thêm vào

def add_embeddings(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]] | None = None,
) -> None:
    """Thêm nhiều vector embedding vào collection """
    if metadatas is None:
        metadatas = [{} for _ in ids]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

def query_embeddings(
    query_embedding: List[float],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Tìm các chunk gần nhất với query_embedding, trả về dict như Chroma default: {ids, distances, documents, metadatas}"""
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return result

def clear_collection() -> None:
    """Xoá toàn bộ dữ liệu trong collection (dùng khi cần reset)"""
    collection.delete(where={})
