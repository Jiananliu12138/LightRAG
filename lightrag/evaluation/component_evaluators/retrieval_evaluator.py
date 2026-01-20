#!/usr/bin/env python3
"""
检索质量评估器 (Retrieval Evaluator)

评估指标:
1. Precision@K - 前K个结果的准确率
2. Recall@K - 前K个结果的召回率
3. MRR (Mean Reciprocal Rank) - 平均倒数排名
4. NDCG (Normalized Discounted Cumulative Gain) - 归一化折损累积增益
5. Hit Rate@K - 命中率
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    precision_at_k: Dict[int, float]  # Precision@K for different K
    recall_at_k: Dict[int, float]     # Recall@K for different K
    mrr: float                         # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]       # NDCG@K for different K
    hit_rate_at_k: Dict[int, float]   # Hit Rate@K
    map_score: float                   # Mean Average Precision
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision@1": round(self.precision_at_k.get(1, 0.0), 4),
            "precision@3": round(self.precision_at_k.get(3, 0.0), 4),
            "precision@5": round(self.precision_at_k.get(5, 0.0), 4),
            "precision@10": round(self.precision_at_k.get(10, 0.0), 4),
            "recall@1": round(self.recall_at_k.get(1, 0.0), 4),
            "recall@3": round(self.recall_at_k.get(3, 0.0), 4),
            "recall@5": round(self.recall_at_k.get(5, 0.0), 4),
            "recall@10": round(self.recall_at_k.get(10, 0.0), 4),
            "mrr": round(self.mrr, 4),
            "ndcg@1": round(self.ndcg_at_k.get(1, 0.0), 4),
            "ndcg@3": round(self.ndcg_at_k.get(3, 0.0), 4),
            "ndcg@5": round(self.ndcg_at_k.get(5, 0.0), 4),
            "ndcg@10": round(self.ndcg_at_k.get(10, 0.0), 4),
            "hit_rate@1": round(self.hit_rate_at_k.get(1, 0.0), 4),
            "hit_rate@3": round(self.hit_rate_at_k.get(3, 0.0), 4),
            "hit_rate@5": round(self.hit_rate_at_k.get(5, 0.0), 4),
            "hit_rate@10": round(self.hit_rate_at_k.get(10, 0.0), 4),
            "map": round(self.map_score, 4),
            "overall_score": round(self._overall_score(), 4)
        }
    
    def _overall_score(self) -> float:
        """计算总体分数"""
        # 综合多个指标
        return (
            self.precision_at_k.get(5, 0.0) * 0.25 +
            self.recall_at_k.get(5, 0.0) * 0.25 +
            self.mrr * 0.25 +
            self.ndcg_at_k.get(5, 0.0) * 0.25
        )


class RetrievalEvaluator:
    """检索质量评估器"""
    
    def __init__(self, retrieval_func: Callable):
        """
        Args:
            retrieval_func: 检索函数，输入查询，返回检索到的文档ID列表
                           示例: async def retrieve(query: str, top_k: int) -> List[str]
        """
        self.retrieval_func = retrieval_func
    
    async def evaluate(
        self,
        test_queries: List[Dict[str, Any]],  # 查询及其相关文档
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """
        评估检索质量
        
        Args:
            test_queries: 测试查询列表，每个查询包含:
                {
                    "query": "查询文本",
                    "relevant_docs": ["doc1", "doc2", ...],  # 相关文档ID列表
                    "relevance_scores": {"doc1": 1.0, "doc2": 0.5, ...}  # 可选：相关度分数
                }
            k_values: 要评估的K值列表
        
        Returns:
            RetrievalMetrics: 评估指标
        """
        print(f"\n{'='*70}")
        print("🔍 检索质量评估")
        print(f"{'='*70}")
        
        max_k = max(k_values)
        
        # 存储所有查询的结果
        all_precisions = defaultdict(list)
        all_recalls = defaultdict(list)
        all_ndcgs = defaultdict(list)
        all_hit_rates = defaultdict(list)
        reciprocal_ranks = []
        average_precisions = []
        
        for idx, test_query in enumerate(test_queries, 1):
            query = test_query["query"]
            relevant_docs = set(test_query["relevant_docs"])
            relevance_scores = test_query.get("relevance_scores", {})
            
            print(f"\n查询 {idx}/{len(test_queries)}: {query[:60]}...")
            
            # 检索文档
            retrieved_docs = await self.retrieval_func(query, max_k)
            
            print(f"  检索到 {len(retrieved_docs)} 个文档")
            print(f"  相关文档数: {len(relevant_docs)}")
            
            # 计算各K值的指标
            for k in k_values:
                retrieved_at_k = retrieved_docs[:k]
                
                # Precision@K
                precision = self._calculate_precision(retrieved_at_k, relevant_docs)
                all_precisions[k].append(precision)
                
                # Recall@K
                recall = self._calculate_recall(retrieved_at_k, relevant_docs)
                all_recalls[k].append(recall)
                
                # NDCG@K
                ndcg = self._calculate_ndcg(retrieved_at_k, relevance_scores, k)
                all_ndcgs[k].append(ndcg)
                
                # Hit Rate@K
                hit_rate = 1.0 if any(doc in relevant_docs for doc in retrieved_at_k) else 0.0
                all_hit_rates[k].append(hit_rate)
            
            # MRR (Mean Reciprocal Rank)
            rr = self._calculate_reciprocal_rank(retrieved_docs, relevant_docs)
            reciprocal_ranks.append(rr)
            
            # MAP (Mean Average Precision)
            ap = self._calculate_average_precision(retrieved_docs, relevant_docs)
            average_precisions.append(ap)
        
        # 计算平均指标
        precision_at_k = {k: np.mean(all_precisions[k]) for k in k_values}
        recall_at_k = {k: np.mean(all_recalls[k]) for k in k_values}
        ndcg_at_k = {k: np.mean(all_ndcgs[k]) for k in k_values}
        hit_rate_at_k = {k: np.mean(all_hit_rates[k]) for k in k_values}
        mrr = np.mean(reciprocal_ranks)
        map_score = np.mean(average_precisions)
        
        metrics = RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_rate_at_k=hit_rate_at_k,
            map_score=map_score
        )
        
        self._display_results(metrics)
        return metrics
    
    def _calculate_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """计算 Precision@K"""
        if not retrieved:
            return 0.0
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(retrieved)
    
    def _calculate_recall(self, retrieved: List[str], relevant: Set[str]) -> float:
        """计算 Recall@K"""
        if not relevant:
            return 0.0
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(relevant)
    
    def _calculate_reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """计算 Reciprocal Rank"""
        for idx, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / idx
        return 0.0
    
    def _calculate_ndcg(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """计算 NDCG@K"""
        if not retrieved:
            return 0.0
        
        # 如果没有提供相关度分数，使用二元相关度（1或0）
        if not relevance_scores:
            relevance_scores = {doc: 1.0 for doc in retrieved}
        
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for idx, doc in enumerate(retrieved[:k], 1):
            relevance = relevance_scores.get(doc, 0.0)
            dcg += relevance / np.log2(idx + 1)
        
        # IDCG (Ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
        
        # NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_average_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """计算 Average Precision"""
        if not relevant:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for idx, doc in enumerate(retrieved, 1):
            if doc in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / idx
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant) if relevant else 0.0
    
    def _display_results(self, metrics: RetrievalMetrics):
        """显示评估结果"""
        print(f"\n{'='*70}")
        print("📈 检索质量指标")
        print(f"{'='*70}")
        
        print(f"\n📊 Precision@K:")
        for k in sorted(metrics.precision_at_k.keys()):
            print(f"  • P@{k:2d}: {metrics.precision_at_k[k]:.2%}")
        
        print(f"\n📊 Recall@K:")
        for k in sorted(metrics.recall_at_k.keys()):
            print(f"  • R@{k:2d}: {metrics.recall_at_k[k]:.2%}")
        
        print(f"\n📊 NDCG@K:")
        for k in sorted(metrics.ndcg_at_k.keys()):
            print(f"  • NDCG@{k:2d}: {metrics.ndcg_at_k[k]:.2%}")
        
        print(f"\n📊 Hit Rate@K:")
        for k in sorted(metrics.hit_rate_at_k.keys()):
            print(f"  • HR@{k:2d}: {metrics.hit_rate_at_k[k]:.2%}")
        
        print(f"\n📊 其他指标:")
        print(f"  • MRR (Mean Reciprocal Rank): {metrics.mrr:.4f}")
        print(f"  • MAP (Mean Average Precision): {metrics.map_score:.4f}")
        
        print(f"\n🎯 总体评分: {metrics._overall_score():.2%}")
        print(f"{'='*70}\n")


# ============================================================================
# 示例使用
# ============================================================================

# 模拟文档库
MOCK_DOCS = {
    "doc1": "LightRAG is a Simple and Fast Retrieval-Augmented Generation framework",
    "doc2": "LightRAG was developed by HKUDS",
    "doc3": "Python is a high-level programming language",
    "doc4": "RAG systems combine retrieval and generation",
    "doc5": "Vector databases store embeddings efficiently",
    "doc6": "Knowledge graphs represent structured information",
    "doc7": "RAGAS is a framework for evaluating RAG systems",
    "doc8": "Ollama runs LLMs locally",
    "doc9": "Embedding models convert text to vectors",
    "doc10": "LLMs can generate human-like text"
}


async def mock_retrieval_func(query: str, top_k: int) -> List[str]:
    """模拟检索函数（用于测试）"""
    # 简单的关键词匹配检索
    query_words = set(query.lower().split())
    
    # 计算每个文档的分数
    scores = {}
    for doc_id, doc_text in MOCK_DOCS.items():
        doc_words = set(doc_text.lower().split())
        # 简单的词重叠分数
        overlap = len(query_words & doc_words)
        scores[doc_id] = overlap
    
    # 按分数排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 返回前 top_k 个文档ID
    return [doc_id for doc_id, score in sorted_docs[:top_k]]


async def test_retrieval_evaluator():
    """测试检索评估器"""
    
    evaluator = RetrievalEvaluator(retrieval_func=mock_retrieval_func)
    
    # 测试查询
    test_queries = [
        {
            "query": "What is LightRAG?",
            "relevant_docs": ["doc1", "doc2"],
            "relevance_scores": {"doc1": 1.0, "doc2": 0.8}
        },
        {
            "query": "How do RAG systems work?",
            "relevant_docs": ["doc1", "doc4", "doc7"],
            "relevance_scores": {"doc1": 0.6, "doc4": 1.0, "doc7": 0.9}
        },
        {
            "query": "Tell me about vector databases",
            "relevant_docs": ["doc5", "doc9"],
            "relevance_scores": {"doc5": 1.0, "doc9": 0.7}
        },
        {
            "query": "What is Python used for?",
            "relevant_docs": ["doc3"],
            "relevance_scores": {"doc3": 1.0}
        },
    ]
    
    # 运行评估
    metrics = await evaluator.evaluate(
        test_queries=test_queries,
        k_values=[1, 3, 5, 10]
    )
    
    print(f"\n完整评估结果:\n{metrics.to_dict()}")


if __name__ == "__main__":
    asyncio.run(test_retrieval_evaluator())
