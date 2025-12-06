from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

# Model embed phổ biến, nhẹ, đủ dùng cho RAG
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_function():
    """ Trả về embedding function dùng cho ChromaDB, Chroma sẽ tự quản lý cache model """
    return SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL_NAME)
