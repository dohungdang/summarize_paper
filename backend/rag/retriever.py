from typing import List
from . import vector_store

def get_relevant_chunks(
    paper_id: str,
    query: str,
    k: int = 5,
) -> List[str]:
    """ Wrapper nhỏ để lấy các đoạn liên quan từ Chroma """
    try:
        return vector_store.query_chunks(paper_id=paper_id, query=query, n_results=k)
    except Exception:
        # nếu vector store lỗi thì trả về list rỗng
        return []
