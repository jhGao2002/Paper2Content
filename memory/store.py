import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document


VECTOR_ROOT = Path(os.getenv("FAISS_INDEX_ROOT", "vectorstores"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    faiss.normalize_L2(vectors)
    return vectors


def _match_filter(metadata: dict, filter_expr: dict | None) -> bool:
    if not filter_expr:
        return True

    must_conditions = filter_expr.get("must", [])
    for cond in must_conditions:
        key = cond.get("key")
        expected = cond.get("match", {}).get("value")
        if metadata.get(key) != expected:
            return False
    return True


class FaissVectorStore:
    def __init__(self, embedding, collection_name: str, persist_root: Path | None = None):
        self.embedding = embedding
        self.collection_name = collection_name
        self.persist_root = (persist_root or VECTOR_ROOT).resolve()
        self.persist_dir = self.persist_root / collection_name
        self.index_path = self.persist_dir / "index.faiss"
        self.docs_path = self.persist_dir / "documents.json"
        self.use_gpu = os.getenv("FAISS_USE_GPU", "1").strip().lower() not in {"0", "false", "no"}
        self.gpu_device = int(os.getenv("FAISS_GPU_DEVICE", "0"))
        self._gpu_resources = None
        self._docs: list[Document] = []
        self._index = None
        self._dim = None
        self._load()

    @property
    def size(self) -> int:
        return len(self._docs)

    def add_documents(self, documents: list[Document]):
        if not documents:
            return

        for start in range(0, len(documents), EMBED_BATCH_SIZE):
            batch_docs = documents[start:start + EMBED_BATCH_SIZE]
            texts = [doc.page_content for doc in batch_docs]
            embeddings = self.embedding.embed_documents(texts)
            vectors = _normalize(np.array(embeddings, dtype="float32"))

            if self._index is None:
                self._dim = vectors.shape[1]
                self._index = self._create_index(self._dim)

            self._index.add(vectors)
            self._docs.extend(deepcopy(batch_docs))

        self._save()

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[Document]:
        return [doc for doc, _score in self.similarity_search_with_score(query, k=k, filter=filter)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        if self._index is None or not self._docs:
            return []

        query_vector = _normalize(np.array(self.embedding.embed_query(query), dtype="float32"))
        fetch_k = len(self._docs) if filter else min(max(k * 5, k), len(self._docs))
        scores, indices = self._index.search(query_vector, fetch_k)

        results: list[tuple[Document, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            doc = self._docs[int(idx)]
            if not _match_filter(doc.metadata, filter):
                continue
            results.append((deepcopy(doc), float(score)))
            if len(results) >= k:
                break
        return results

    def list_sources(self) -> list[str]:
        return sorted({doc.metadata.get("source", "") for doc in self._docs if doc.metadata.get("source")})

    def _create_index(self, dim: int):
        cpu_index = faiss.IndexFlatIP(dim)
        if not self.use_gpu:
            return cpu_index

        try:
            self._gpu_resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(self._gpu_resources, self.gpu_device, cpu_index)
        except Exception as exc:
            print(f"[FAISS] GPU 初始化失败，回退 CPU: {exc}")
            return cpu_index

    def _to_cpu_index(self):
        if self._index is None:
            return None
        try:
            return faiss.index_gpu_to_cpu(self._index)
        except Exception:
            return self._index

    def _save(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        cpu_index = self._to_cpu_index()
        if cpu_index is not None:
            faiss.write_index(cpu_index, str(self.index_path))

        payload = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self._docs
        ]
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load(self):
        if self.docs_path.exists():
            with open(self.docs_path, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
            self._docs = [
                Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
                for item in raw_docs
            ]

        if self.index_path.exists():
            cpu_index = faiss.read_index(str(self.index_path))
            self._dim = cpu_index.d
            if self.use_gpu:
                try:
                    self._gpu_resources = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(self._gpu_resources, self.gpu_device, cpu_index)
                except Exception as exc:
                    print(f"[FAISS] 索引加载到 GPU 失败，回退 CPU: {exc}")
                    self._index = cpu_index
            else:
                self._index = cpu_index


def init_vector_stores(embeddings) -> tuple[None, FaissVectorStore, FaissVectorStore]:
    pdf_store = FaissVectorStore(embedding=embeddings, collection_name="pdf_knowledge")
    memory_store = FaissVectorStore(embedding=embeddings, collection_name="user_semantic_memory")
    return None, pdf_store, memory_store


def build_memory_context(memory_store: FaissVectorStore, user_id: str, session_id: str, question: str) -> str:
    try:
        results = memory_store.similarity_search_with_score(
            question,
            k=10,
            filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "session_id", "match": {"value": session_id}},
                ]
            },
        )
    except Exception:
        results = []

    if not results:
        return ""

    now = datetime.now()
    scored = []
    for doc, similarity in results:
        ts = doc.metadata.get("timestamp", "")
        try:
            days_ago = (now - datetime.fromisoformat(ts)).days
        except Exception:
            days_ago = 30
        recency = 1 / (1 + days_ago)
        score = 0.7 * similarity + 0.3 * recency
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored[:3] if score >= 0.5]

    if not top_docs:
        return ""

    facts = [d for d in top_docs if d.metadata.get("type") == "auto_fact"]
    notes = [d for d in top_docs if d.metadata.get("type") == "note"]

    parts = ["【本会话的长期记忆】："]
    if facts:
        parts.append("[事实]")
        parts.extend(f"- {d.page_content}" for d in facts)
    if notes:
        parts.append("[笔记]")
        parts.extend(f"- {d.page_content}" for d in notes)
    parts.append("请在回答时参考以上背景。")

    return "\n".join(parts)


def save_fact(memory_store: FaissVectorStore, content: str, user_id: str, session_id: str, fact_type: str = "auto_fact"):
    doc = Document(
        page_content=content,
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "type": fact_type,
            "timestamp": datetime.now().isoformat(),
        },
    )
    memory_store.add_documents([doc])
