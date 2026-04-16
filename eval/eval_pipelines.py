from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document

from config import get_embeddings
from document.chunking import split_text
from memory.store import FaissVectorStore


def build_answer_prompt(question: str, context: str) -> str:
    return (
        "你是学术论文问答助手。请严格基于给定上下文回答问题，不要编造上下文中不存在的信息。\n"
        "如果上下文不足以支持回答，请明确说明“上下文不足，无法确定”。\n\n"
        f"问题：{question}\n\n"
        f"上下文：\n{context}"
    )


class FlatChunkRAG:
    def __init__(self, store: FaissVectorStore, llm):
        self.store = store
        self.llm = llm

    def load_document(self, pdf_path: Path) -> dict:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        chunks = split_text(md_text, chunk_size=800, chunk_overlap=150)
        docs = [
            Document(page_content=chunk.page_content, metadata={"source": pdf_path.name})
            for chunk in chunks
        ]
        self.store.add_documents(docs)
        return {"document": pdf_path.name, "chunk_count": len(docs)}

    def ask(self, question: str) -> tuple[str, str]:
        docs = self.store.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self.llm.invoke(build_answer_prompt(question, context)).content
        return str(answer).strip(), context


class ParentChildRAG:
    def __init__(self, store: FaissVectorStore, llm):
        self.store = store
        self.llm = llm

    def load_document(self, pdf_path: Path) -> dict:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        parent_docs = split_text(md_text, chunk_size=1500, chunk_overlap=200)
        child_docs = []

        for parent_index, parent_doc in enumerate(parent_docs):
            parent_id = f"{pdf_path.name}::parent::{parent_index}"
            parent_text = parent_doc.page_content
            for child_doc in split_text(parent_text, chunk_size=400, chunk_overlap=50):
                child_doc.metadata = {
                    "source": pdf_path.name,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                }
                child_docs.append(child_doc)

        self.store.add_documents(child_docs)
        return {"document": pdf_path.name, "chunk_count": len(child_docs)}

    def ask(self, question: str) -> tuple[str, str]:
        child_docs = self.store.similarity_search(question, k=8)
        parent_map: dict[str, str] = {}
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id", "")
            if parent_id and parent_id not in parent_map:
                parent_map[parent_id] = doc.metadata.get("parent_text", doc.page_content)
            if len(parent_map) >= 4:
                break

        context = "\n\n".join(parent_map.values())
        answer = self.llm.invoke(build_answer_prompt(question, context)).content
        return str(answer).strip(), context


class LexicalHashEmbeddings:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in self._tokenize(text):
            bucket = hash(token) % self.dim
            vector[bucket] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(model_name: str):
    if model_name == "Embedding-3":
        return get_embeddings(), "Embedding-3"
    return LexicalHashEmbeddings(), "LexicalHashEmbedding"


def build_pipeline(use_parent_child: bool, store: FaissVectorStore, llm):
    if use_parent_child:
        return ParentChildRAG(store, llm)
    return FlatChunkRAG(store, llm)
