"""
RAG 组件评估器模块 (基于学术研究)

包含：
- ChunkingEvaluator: 分块质量评估 (Semantic Cohesion, Information Gain, etc.)
- EmbeddingEvaluator: 嵌入质量评估 (K-NN Consistency, Semantic Overlap, etc.)
- RetrievalEvaluator: 检索质量评估 (NDCG, MRR, Hit Rate, etc.)
- RerankingEvaluator: 重排质量评估 (Precision Gain, NDCG Improvement, etc.)

References:
- ACL 2019: Text Segmentation by Cross-Lingual Word Embeddings
- Järvelin & Kekäläinen 2002: Cumulative gain-based evaluation
- Liu 2009: Learning to Rank for Information Retrieval
- LightRAG Paper 2024: Graph-based RAG Evaluation
"""

from .chunking_evaluator import ChunkingEvaluator, ChunkingMetrics
from .embedding_evaluator import EmbeddingEvaluator, EmbeddingMetrics
from .retrieval_evaluator import RetrievalEvaluator, RetrievalMetrics
from .reranking_evaluator import RerankingEvaluator, RerankingMetrics

__all__ = [
    'ChunkingEvaluator',
    'ChunkingMetrics',
    'EmbeddingEvaluator',
    'EmbeddingMetrics',
    'RetrievalEvaluator',
    'RetrievalMetrics',
    'RerankingEvaluator',
    'RerankingMetrics',
]
