from typing import List

def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 200,
) -> List[str]:
    """ Tách text dài thành các đoạn nhỏ hơn để dùng cho RAG, ở đây chỉ dùng heuristic 
    đơn giản theo số lượng ký tự, tránh cắt đúng giữa câu nếu được."""
    text = (text or "").strip()
    if not text:
        return []
    text = _normalize_whitespace(text)
    n = len(text)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        # Cố gắng ngắt ở dấu chấm gần cuối
        if end < n:
            last_dot = chunk.rfind(".")
            if last_dot != -1 and last_dot > max_chars * 0.4:
                end = start + last_dot + 1
                chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # overlap để giữ ngữ cảnh
        start = max(0, end - overlap)
    return chunks
