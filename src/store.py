from __future__ import annotations

from typing import Any, Callable
import chromadb
from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed, _local_embedd
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed or _local_embedd
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:

            # Khởi tạo ephemeral Client (in-memory) để mỗi instance bắt đầu trống
            client = chromadb.Client()
            unique_name = f"{self._collection_name}_{id(self)}"
            self._collection = client.get_or_create_collection(unique_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_vec = self._embedding_fn(query)
        results = []
        for record in records:
            score = compute_similarity(query_vec, record["embedding"])
            results.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "score": score,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        if self._use_chroma:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                ids.append(doc.id)
                documents.append(doc.content)
                embeddings.append(self._embedding_fn(doc.content))
                metadatas.append(doc.metadata if doc.metadata else {"_empty_": True})
            if ids:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            res = self._collection.query(query_embeddings=[query_vec], n_results=top_k)
            results = []
            if res and "ids" in res and res["ids"] and len(res["ids"][0]) > 0:
                for i in range(len(res["ids"][0])):
                    results.append(
                        {
                            "id": res["ids"][0][i],
                            "content": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i],
                            "score": (
                                1.0 / (1.0 + res["distances"][0][i])
                                if "distances" in res and res["distances"]
                                else 0.0
                            ),
                        }
                    )
            return results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""

        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict = None
    ) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            res = self._collection.query(
                query_embeddings=[query_vec], n_results=top_k, where=metadata_filter
            )
            results = []
            if res and "ids" in res and res["ids"] and len(res["ids"][0]) > 0:
                for i in range(len(res["ids"][0])):
                    results.append(
                        {
                            "id": res["ids"][0][i],
                            "content": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i],
                            "score": (
                                1.0 / (1.0 + res["distances"][0][i])
                                if "distances" in res and res["distances"]
                                else 0.0
                            ),
                        }
                    )
            return results
        else:
            if not metadata_filter:
                filtered_records = self._store
            else:
                filtered_records = []
                for record in self._store:
                    match = True
                    for k, v in metadata_filter.items():
                        if record["metadata"].get(k) != v:
                            match = False
                            break
                    if match:
                        filtered_records.append(record)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma:
            initial_count = self._collection.count()
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass
            return self._collection.count() < initial_count
        else:
            initial_len = len(self._store)
            self._store = [
                r
                for r in self._store
                if r.get("metadata", {}).get("doc_id") != doc_id
                and r.get("id") != doc_id
            ]
            return len(self._store) < initial_len
