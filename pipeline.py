import os

from src.config import PDF_PATH, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, CHAT_MODEL
from src.loader import load_pdf
from src.chunker import chunk
from src.vectorstore import vectorize, load_vectorstore
from src.retriever import retrieve
from src.generator import build_chain


def build_index():
    docs = load_pdf(PDF_PATH)
    chunks = chunk(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    vectorize(chunks, CHROMA_DIR)


def get_chain():
    if not os.path.exists(CHROMA_DIR):
        build_index()
    vectorstore = load_vectorstore(CHROMA_DIR)
    retriever = retrieve(vectorstore, TOP_K)
    return build_chain(retriever, CHAT_MODEL)


if __name__ == "__main__":
    chain = get_chain()
    print(chain.invoke("What is self-attention?"))