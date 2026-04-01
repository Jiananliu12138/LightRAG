from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MilvusLiteConfig:
    uri: str | None = "/data/h50056789/LightRAG/workspace/rag_storage_milvus/LightRAG/LightRAG.db"
    index_type: str = "AUTOINDEX"
    enable_sparse: bool = True
    enable_hybrid_search: bool = True
    hybrid_ranker: str = "RRFRanker"
    hybrid_ranker_k: int = 60
    
    dense_weight: float = 1.0
    sparse_weight: float = 0.35
    sparse_max_features: int = 256
    sparse_min_token_length: int = 2
    candidate_limit_multiplier: int = 4


def configure_local_milvus_lite(
    working_dir: str | Path,
    config: MilvusLiteConfig,
) -> Path:
    working_path = Path(working_dir).resolve()
    db_path = Path(config.uri).resolve() if config.uri else working_path / "milvus_lite.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["MILVUS_LITE_URI"] = str(db_path)
    os.environ.setdefault("MILVUS_URI", "http://127.0.0.1:19530")
    os.environ["MILVUS_DB_NAME"] = ""
    os.environ["MILVUS_ENABLE_SPARSE"] = str(config.enable_sparse).lower()
    os.environ["MILVUS_ENABLE_HYBRID_SEARCH"] = str(
        config.enable_hybrid_search
    ).lower()
    os.environ["MILVUS_HYBRID_RANKER"] = config.hybrid_ranker
    os.environ["MILVUS_HYBRID_RANKER_K"] = str(config.hybrid_ranker_k)
    os.environ["MILVUS_HYBRID_DENSE_WEIGHT"] = str(config.dense_weight)
    os.environ["MILVUS_HYBRID_SPARSE_WEIGHT"] = str(config.sparse_weight)
    os.environ["MILVUS_HYBRID_SPARSE_MAX_FEATURES"] = str(
        config.sparse_max_features
    )
    os.environ["MILVUS_HYBRID_SPARSE_MIN_TOKEN_LENGTH"] = str(
        config.sparse_min_token_length
    )
    os.environ["MILVUS_HYBRID_CANDIDATE_LIMIT_MULTIPLIER"] = str(
        config.candidate_limit_multiplier
    )

    return db_path


def build_milvus_vector_storage_kwargs(config: MilvusLiteConfig) -> dict[str, Any]:
    return {
        "index_type": config.index_type,
        "enable_sparse": config.enable_sparse,
        "enable_hybrid_search": config.enable_hybrid_search,
        "hybrid_ranker": config.hybrid_ranker,
        "hybrid_ranker_k": config.hybrid_ranker_k,
        "dense_weight": config.dense_weight,
        "sparse_weight": config.sparse_weight,
        "sparse_max_features": config.sparse_max_features,
        "sparse_min_token_length": config.sparse_min_token_length,
        "candidate_limit_multiplier": config.candidate_limit_multiplier,
    }
