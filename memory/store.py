import json
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import faiss
import numpy as np
from langchain_core.documents import Document


VECTOR_ROOT = Path(os.getenv("FAISS_INDEX_ROOT", "vectorstores"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
_GPU_CAPABILITY_WARNED = False


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
        match = cond.get("match", {})
        if "value" in match and metadata.get(key) != match["value"]:
            return False
        if "any" in match and metadata.get(key) not in set(match["any"]):
            return False
    return True


def _filter_values(filter_expr: dict | None, key: str) -> list[str]:
    if not filter_expr:
        return []

    values: list[str] = []
    for cond in filter_expr.get("must", []):
        if cond.get("key") != key:
            continue
        match = cond.get("match", {})
        if "value" in match:
            values.append(str(match["value"]))
        elif "any" in match:
            values.extend(str(item) for item in match["any"])
    return list(dict.fromkeys(values))


def _faiss_gpu_available() -> bool:
    return all(
        hasattr(faiss, attr)
        for attr in ("StandardGpuResources", "index_cpu_to_gpu", "index_gpu_to_cpu")
    )


class FaissVectorStore:
    def __init__(
        self,
        embedding,
        collection_name: str,
        persist_root: Path | None = None,
        shard_key: str | None = None,
    ):
        self.embedding = embedding
        self.collection_name = collection_name
        self.persist_root = (persist_root or VECTOR_ROOT).resolve()
        self.persist_dir = self.persist_root / collection_name
        self.index_path = self.persist_dir / "index.faiss"
        self.docs_path = self.persist_dir / "documents.json"
        self.shard_key = shard_key
        requested_gpu = os.getenv("FAISS_USE_GPU", "1").strip().lower() not in {"0", "false", "no"}
        self.use_gpu = requested_gpu and _faiss_gpu_available()
        self.gpu_device = int(os.getenv("FAISS_GPU_DEVICE", "0"))
        self._gpu_resources = None
        self._docs: list[Document] = []
        self._index = None
        self._dim = None
        self._shards: dict[str, dict] = {}

        global _GPU_CAPABILITY_WARNED
        if requested_gpu and not self.use_gpu and not _GPU_CAPABILITY_WARNED:
            print("[FAISS] GPU FAISS is unavailable; falling back to CPU.")
            _GPU_CAPABILITY_WARNED = True

        if self.shard_key:
            self._load_shards()
        else:
            self._load()

    @property
    def size(self) -> int:
        if self.shard_key:
            return sum(len(shard["docs"]) for shard in self._shards.values())
        return len(self._docs)

    def add_documents(self, documents: list[Document]):
        if not documents:
            return

        if self.shard_key:
            grouped: dict[str, list[Document]] = {}
            for doc in documents:
                shard_value = str(doc.metadata.get(self.shard_key, "")).strip()
                if not shard_value:
                    raise ValueError(f"sharded vector store requires metadata[{self.shard_key!r}]")
                grouped.setdefault(shard_value, []).append(doc)
            for shard_value, shard_docs in grouped.items():
                self._add_documents_to_shard(shard_value, shard_docs)
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
        if self.shard_key:
            return self._similarity_search_shards(query, k=k, filter=filter)

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

    def rank_sources_by_text(
        self,
        query: str,
        source_texts: dict[str, str],
        top_n: int = 5,
    ) -> list[str]:
        candidates = [
            (source, text.strip())
            for source, text in source_texts.items()
            if source and text and self._has_shard(source)
        ]
        if not candidates:
            return []

        query_vector = _normalize(np.array(self.embedding.embed_query(query), dtype="float32"))
        text_vectors = _normalize(
            np.array(self.embedding.embed_documents([text for _, text in candidates]), dtype="float32")
        )
        scores = np.dot(text_vectors, query_vector[0])
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [source for (source, _text), _score in ranked[:top_n]]

    def list_sources(self) -> list[str]:
        if self.shard_key:
            return sorted(self._shards.keys())
        return sorted({doc.metadata.get("source", "") for doc in self._docs if doc.metadata.get("source")})

    def get_documents(self, filter: dict | None = None, limit: int | None = None) -> list[Document]:
        if self.shard_key:
            docs = []
            for shard_value in self._selected_shard_values(filter):
                shard = self._shards.get(shard_value)
                if not shard:
                    continue
                for doc in shard["docs"]:
                    if not _match_filter(doc.metadata, filter):
                        continue
                    docs.append(deepcopy(doc))
                    if limit is not None and len(docs) >= limit:
                        return docs
            return docs

        results: list[Document] = []
        for doc in self._docs:
            if not _match_filter(doc.metadata, filter):
                continue
            results.append(deepcopy(doc))
            if limit is not None and len(results) >= limit:
                break
        return results

    def delete_documents(self, filter: dict | None = None) -> int:
        if self.shard_key:
            return self._delete_shard_documents(filter)

        if not self._docs:
            return 0

        remaining_docs = [doc for doc in self._docs if not _match_filter(doc.metadata, filter)]
        deleted_count = len(self._docs) - len(remaining_docs)
        if deleted_count == 0:
            return 0

        self._docs = [deepcopy(doc) for doc in remaining_docs]
        self._rebuild_index()
        self._save()
        return deleted_count

    def _add_documents_to_shard(self, shard_value: str, documents: list[Document]):
        shard = self._shards.get(shard_value)
        if shard is None:
            shard = {"docs": [], "index": None, "dim": None, "gpu_resources": None}
            self._shards[shard_value] = shard

        for start in range(0, len(documents), EMBED_BATCH_SIZE):
            batch_docs = documents[start:start + EMBED_BATCH_SIZE]
            texts = [doc.page_content for doc in batch_docs]
            embeddings = self.embedding.embed_documents(texts)
            vectors = _normalize(np.array(embeddings, dtype="float32"))

            if shard["index"] is None:
                shard["dim"] = vectors.shape[1]
                shard["index"] = self._create_index(shard["dim"])

            shard["index"].add(vectors)
            shard["docs"].extend(deepcopy(batch_docs))

        self._save_shard(shard_value)

    def _similarity_search_shards(
        self,
        query: str,
        k: int,
        filter: dict | None,
    ) -> list[tuple[Document, float]]:
        if not self._shards:
            return []

        query_vector = _normalize(np.array(self.embedding.embed_query(query), dtype="float32"))
        results: list[tuple[Document, float]] = []
        for shard_value in self._selected_shard_values(filter):
            shard = self._shards.get(shard_value)
            if not shard or shard["index"] is None or not shard["docs"]:
                continue

            fetch_k = min(max(k * 5, k), len(shard["docs"]))
            scores, indices = shard["index"].search(query_vector, fetch_k)
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                doc = shard["docs"][int(idx)]
                if not _match_filter(doc.metadata, filter):
                    continue
                results.append((deepcopy(doc), float(score)))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    def _selected_shard_values(self, filter: dict | None) -> list[str]:
        if not self.shard_key:
            return []
        selected = _filter_values(filter, self.shard_key)
        if selected:
            return [value for value in selected if value in self._shards]
        return sorted(self._shards.keys())

    def _delete_shard_documents(self, filter: dict | None) -> int:
        if not self._shards:
            return 0

        selected = _filter_values(filter, self.shard_key)
        if selected:
            deleted = 0
            for shard_value in selected:
                shard = self._shards.pop(shard_value, None)
                if not shard:
                    continue
                deleted += len(shard["docs"])
                shutil.rmtree(self._shard_dir(shard_value), ignore_errors=True)
            return deleted

        deleted = 0
        for shard_value in list(self._shards):
            shard = self._shards[shard_value]
            remaining_docs = [doc for doc in shard["docs"] if not _match_filter(doc.metadata, filter)]
            deleted += len(shard["docs"]) - len(remaining_docs)
            if not remaining_docs:
                self._shards.pop(shard_value, None)
                shutil.rmtree(self._shard_dir(shard_value), ignore_errors=True)
                continue
            shard["docs"] = [deepcopy(doc) for doc in remaining_docs]
            self._rebuild_shard_index(shard_value)
            self._save_shard(shard_value)
        return deleted

    def _create_index(self, dim: int):
        cpu_index = faiss.IndexFlatIP(dim)
        if not self.use_gpu:
            return cpu_index

        try:
            resources = faiss.StandardGpuResources()
            self._gpu_resources = resources
            return faiss.index_cpu_to_gpu(resources, self.gpu_device, cpu_index)
        except Exception as exc:
            print(f"[FAISS] Could not create GPU index; falling back to CPU: {exc}")
            return cpu_index

    def _to_cpu_index(self, index=None):
        index = self._index if index is None else index
        if index is None:
            return None
        if not hasattr(faiss, "index_gpu_to_cpu"):
            return index
        try:
            return faiss.index_gpu_to_cpu(index)
        except Exception:
            return index

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

    def _save_shard(self, shard_value: str):
        shard = self._shards[shard_value]
        shard_dir = self._shard_dir(shard_value)
        shard_dir.mkdir(parents=True, exist_ok=True)

        cpu_index = self._to_cpu_index(shard["index"])
        if cpu_index is not None:
            faiss.write_index(cpu_index, str(shard_dir / "index.faiss"))

        payload = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in shard["docs"]
        ]
        with open(shard_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _rebuild_index(self):
        self._index = None
        self._dim = None
        self._gpu_resources = None
        if not self._docs:
            return

        for start in range(0, len(self._docs), EMBED_BATCH_SIZE):
            batch_docs = self._docs[start:start + EMBED_BATCH_SIZE]
            texts = [doc.page_content for doc in batch_docs]
            embeddings = self.embedding.embed_documents(texts)
            vectors = _normalize(np.array(embeddings, dtype="float32"))

            if self._index is None:
                self._dim = vectors.shape[1]
                self._index = self._create_index(self._dim)

            self._index.add(vectors)

    def _rebuild_shard_index(self, shard_value: str):
        shard = self._shards[shard_value]
        shard["index"] = None
        shard["dim"] = None
        shard["gpu_resources"] = None
        if not shard["docs"]:
            return

        for start in range(0, len(shard["docs"]), EMBED_BATCH_SIZE):
            batch_docs = shard["docs"][start:start + EMBED_BATCH_SIZE]
            texts = [doc.page_content for doc in batch_docs]
            embeddings = self.embedding.embed_documents(texts)
            vectors = _normalize(np.array(embeddings, dtype="float32"))

            if shard["index"] is None:
                shard["dim"] = vectors.shape[1]
                shard["index"] = self._create_index(shard["dim"])

            shard["index"].add(vectors)

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
                    print(f"[FAISS] Could not load GPU index; falling back to CPU: {exc}")
                    self._index = cpu_index
            else:
                self._index = cpu_index

    def _load_shards(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_root_index()

        for shard_dir in self.persist_dir.iterdir():
            if not shard_dir.is_dir():
                continue
            docs_path = shard_dir / "documents.json"
            index_path = shard_dir / "index.faiss"
            if not docs_path.exists() or not index_path.exists():
                continue

            with open(docs_path, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
            docs = [
                Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
                for item in raw_docs
            ]
            if not docs:
                continue
            shard_value = str(docs[0].metadata.get(self.shard_key, shard_dir.name))
            cpu_index = faiss.read_index(str(index_path))
            index = cpu_index
            if self.use_gpu:
                try:
                    resources = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(resources, self.gpu_device, cpu_index)
                except Exception as exc:
                    print(f"[FAISS] Could not load GPU shard index; falling back to CPU: {exc}")
            self._shards[shard_value] = {
                "docs": docs,
                "index": index,
                "dim": cpu_index.d,
                "gpu_resources": None,
            }

    def _migrate_legacy_root_index(self):
        if not self.docs_path.exists() or not self.index_path.exists():
            self._remove_legacy_root_files()
            return

        with open(self.docs_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        docs = [
            Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
            for item in raw_docs
        ]
        cpu_index = faiss.read_index(str(self.index_path))

        grouped: dict[str, list[tuple[Document, np.ndarray]]] = {}
        for idx, doc in enumerate(docs):
            shard_value = str(doc.metadata.get(self.shard_key, "")).strip()
            if not shard_value:
                continue
            vector = np.zeros(cpu_index.d, dtype="float32")
            cpu_index.reconstruct(idx, vector)
            grouped.setdefault(shard_value, []).append((doc, vector))

        for shard_value, pairs in grouped.items():
            shard_index = faiss.IndexFlatIP(cpu_index.d)
            vectors = np.array([vector for _doc, vector in pairs], dtype="float32")
            shard_index.add(vectors)
            self._shards[shard_value] = {
                "docs": [deepcopy(doc) for doc, _vector in pairs],
                "index": shard_index,
                "dim": cpu_index.d,
                "gpu_resources": None,
            }
            self._save_shard(shard_value)

        self._remove_legacy_root_files()

    def _remove_legacy_root_files(self):
        for path in (self.index_path, self.docs_path):
            if path.exists():
                path.unlink()

    def _has_shard(self, shard_value: str) -> bool:
        return shard_value in self._shards

    def _shard_dir(self, shard_value: str) -> Path:
        return self.persist_dir / quote(shard_value, safe="")


def init_vector_stores(embeddings) -> tuple[None, FaissVectorStore, FaissVectorStore]:
    pdf_store = FaissVectorStore(embedding=embeddings, collection_name="pdf_knowledge", shard_key="source")
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

    parts = ["Relevant user memory:"]
    if facts:
        parts.append("[Facts]")
        parts.extend(f"- {d.page_content}" for d in facts)
    if notes:
        parts.append("[Notes]")
        parts.extend(f"- {d.page_content}" for d in notes)
    parts.append("Use these memories only when they are relevant to the current question.")

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
