import importlib.util
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from pymilvus import RRFRanker

from lightrag.kg.milvus_impl import MilvusVectorDBStorage


pytestmark = pytest.mark.offline


class DummyEmbeddingFunc:
    embedding_dim = 2
    model_name = "dummy-embed"

    async def __call__(self, texts: list[str], _priority: int | None = None):
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.lower()
            if "apple" in normalized:
                vectors.append([1.0, 0.0])
            elif "banana" in normalized:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return np.array(vectors, dtype=np.float32)


def test_local_milvus_uri_ignores_db_name(tmp_path, monkeypatch):
    monkeypatch.setenv("MILVUS_LITE_URI", str(tmp_path / "milvus_lite.db"))
    monkeypatch.setenv("MILVUS_DB_NAME", "should_be_ignored")

    storage = MilvusVectorDBStorage(
        namespace="chunks",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 8,
            "working_dir": str(tmp_path),
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
                "enable_sparse": True,
                "enable_hybrid_search": True,
                "hybrid_ranker": "RRFRanker",
                "hybrid_ranker_k": 60,
            },
        },
        embedding_func=DummyEmbeddingFunc(),
        meta_fields={"content", "full_doc_id"},
    )

    assert storage._get_milvus_db_name() is None
    assert storage.hybrid_search_config.enable_sparse is True
    assert storage.hybrid_search_config.enable_hybrid_search is True
    assert storage.hybrid_search_config.hybrid_ranker == "RRFRanker"


def test_rrf_ranker_is_selected(tmp_path):
    storage = MilvusVectorDBStorage(
        namespace="chunks",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 8,
            "working_dir": str(tmp_path),
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
                "enable_sparse": True,
                "enable_hybrid_search": True,
                "hybrid_ranker": "RRFRanker",
                "hybrid_ranker_k": 60,
            },
        },
        embedding_func=DummyEmbeddingFunc(),
        meta_fields={"content", "full_doc_id"},
    )

    ranker = storage._build_hybrid_ranker()
    assert isinstance(ranker, RRFRanker)


@pytest.mark.asyncio
async def test_sparse_vectors_are_persisted_when_enabled(tmp_path):
    storage = MilvusVectorDBStorage(
        namespace="chunks",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 8,
            "working_dir": str(tmp_path),
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
                "enable_sparse": True,
                "enable_hybrid_search": False,
            },
        },
        embedding_func=DummyEmbeddingFunc(),
        meta_fields={"content", "full_doc_id"},
    )
    storage._client = MagicMock()
    storage.final_namespace = "test_workspace_chunks"
    storage._ensure_collection_loaded = MagicMock()

    await storage.upsert(
        {"chunk-1": {"content": "banana market reference", "full_doc_id": "doc-1"}}
    )

    inserted_rows = storage._client.upsert.call_args.kwargs["data"]
    assert "sparse_vector" in inserted_rows[0]
    assert inserted_rows[0]["sparse_vector"]


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Milvus Lite is not supported natively on Windows",
)
@pytest.mark.skipif(
    importlib.util.find_spec("milvus_lite") is None,
    reason="milvus_lite runtime is not installed",
)
@pytest.mark.asyncio
async def test_milvus_lite_hybrid_search_prefers_sparse_match(tmp_path, monkeypatch):
    monkeypatch.setenv("MILVUS_LITE_URI", str(tmp_path / "milvus_lite.db"))
    monkeypatch.setenv("MILVUS_DB_NAME", "")

    storage = MilvusVectorDBStorage(
        namespace="chunks",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 8,
            "working_dir": str(tmp_path),
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
                "enable_sparse": True,
                "enable_hybrid_search": True,
                "hybrid_ranker": "RRFRanker",
                "hybrid_ranker_k": 60,
                "candidate_limit_multiplier": 2,
            },
        },
        embedding_func=DummyEmbeddingFunc(),
        meta_fields={"content", "full_doc_id"},
    )

    await storage.initialize()
    await storage.upsert(
        {
            "chunk-apple": {
                "content": "apple orchard reference",
                "full_doc_id": "doc-apple",
            },
            "chunk-banana": {
                "content": "banana market reference",
                "full_doc_id": "doc-banana",
            },
        }
    )

    results = await storage.query(
        "banana banana",
        top_k=1,
        query_embedding=[1.0, 0.0],
    )

    assert results
    assert results[0]["id"] == "chunk-banana"
