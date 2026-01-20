#!/usr/bin/env python3
"""
重排质量评估器 (Reranking Evaluator)

基于学术研究的重排评估指标:
1. Precision Gain - 重排后精确度提升
2. nDCG Improvement - NDCG指标改进
3. MRR Improvement - MRR指标改进
4. Latency Cost - 延迟成本
5. Signal-to-Noise Ratio - 信噪比改进

References:
- "Learning to Rank for Information Retrieval" (Liu 2009)
- "Pretrained Transformers for Text Ranking" (Nogueira & Cho 2020)
- "RankGPT: LLMs as Re-Ranking Agent" (Sun et al. 2023)
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RerankingMetrics:
    """
    重排评估指标
    
    References:
    - Precision Gain: "Learning to Rank" (Liu 2009)
    - nDCG: "Cumulative Gain-based Evaluation" (Järvelin & Kekäläinen 2002)
    - MRR: Classical IR metric
    """
    precision_gain_at_k: Dict[int, float]      # 精确度提升 @K
    ndcg_improvement_at_k: Dict[int, float]    # NDCG改进 @K
    mrr_improvement: float                      # MRR改进
    signal_to_noise_improvement: float          # 信噪比改进
    avg_latency_ms: float                       # 平均延迟(毫秒)
    latency_quality_ratio: float                # 延迟-质量比
    total_queries: int                          # 总查询数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision_gain@1": round(self.precision_gain_at_k.get(1, 0.0), 4),
            "precision_gain@3": round(self.precision_gain_at_k.get(3, 0.0), 4),
            "precision_gain@5": round(self.precision_gain_at_k.get(5, 0.0), 4),
            "ndcg_improvement@1": round(self.ndcg_improvement_at_k.get(1, 0.0), 4),
            "ndcg_improvement@3": round(self.ndcg_improvement_at_k.get(3, 0.0), 4),
            "ndcg_improvement@5": round(self.ndcg_improvement_at_k.get(5, 0.0), 4),
            "mrr_improvement": round(self.mrr_improvement, 4),
            "signal_to_noise_improvement": round(self.signal_to_noise_improvement, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "latency_quality_ratio": round(self.latency_quality_ratio, 4),
            "total_queries": self.total_queries,
            "overall_score": round(self._overall_score(), 4)
        }
    
    def _overall_score(self) -> float:
        """
        计算总体评分
        
        权重分配 (基于实际应用重要性):
        - NDCG Improvement @3: 40%
        - Precision Gain @3: 30%
        - MRR Improvement: 20%
        - Signal-to-Noise: 10%
        """
        return (
            self.ndcg_improvement_at_k.get(3, 0.0) * 0.40 +
            self.precision_gain_at_k.get(3, 0.0) * 0.30 +
            max(self.mrr_improvement, 0.0) * 0.20 +  # MRR可能为负
            max(self.signal_to_noise_improvement, 0.0) * 0.10
        )


class RerankingEvaluator:
    """
    重排质量评估器
    
    评估重排模型对初始检索结果的改进程度
    """
    
    def __init__(
        self,
        initial_retrieval_func: Callable,  # 初始检索函数
        reranking_func: Callable            # 重排函数
    ):
        """
        Args:
            initial_retrieval_func: async def(query: str, top_k: int) -> List[str]
            reranking_func: async def(query: str, doc_ids: List[str]) -> List[str]
        """
        self.initial_retrieval_func = initial_retrieval_func
        self.reranking_func = reranking_func
    
    async def evaluate(
        self,
        test_queries: List[Dict[str, Any]],  # 测试查询
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RerankingMetrics:
        """
        评估重排质量
        
        Args:
            test_queries: 测试查询列表，格式:
                {
                    "query": "查询文本",
                    "relevant_docs": ["doc1", "doc2", ...],
                    "relevance_scores": {"doc1": 1.0, "doc2": 0.5, ...}
                }
            k_values: 要评估的K值列表
        
        Returns:
            RerankingMetrics: 评估指标
        """
        print(f"\n{'='*70}")
        print("🔄 重排质量评估 (Reranking Evaluation)")
        print(f"{'='*70}")
        
        max_k = max(k_values)
        
        # 存储各指标
        precision_before = defaultdict(list)
        precision_after = defaultdict(list)
        ndcg_before = defaultdict(list)
        ndcg_after = defaultdict(list)
        mrr_before_list = []
        mrr_after_list = []
        snr_before_list = []
        snr_after_list = []
        latencies = []
        
        for idx, test_query in enumerate(test_queries, 1):
            query = test_query["query"]
            relevant_docs = set(test_query["relevant_docs"])
            relevance_scores = test_query.get("relevance_scores", {})
            
            print(f"\n查询 {idx}/{len(test_queries)}: {query[:50]}...")
            
            # 1. 初始检索
            initial_results = await self.initial_retrieval_func(query, max_k)
            
            # 2. 重排 (计时)
            start_time = time.time()
            reranked_results = await self.reranking_func(query, initial_results)
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            latencies.append(latency)
            
            print(f"  初始检索: {len(initial_results)} 个文档")
            print(f"  重排延迟: {latency:.2f}ms")
            
            # 3. 计算各K值的指标
            for k in k_values:
                # Precision
                p_before = self._calculate_precision(initial_results[:k], relevant_docs)
                p_after = self._calculate_precision(reranked_results[:k], relevant_docs)
                precision_before[k].append(p_before)
                precision_after[k].append(p_after)
                
                # NDCG
                ndcg_before_k = self._calculate_ndcg(initial_results[:k], relevance_scores, k)
                ndcg_after_k = self._calculate_ndcg(reranked_results[:k], relevance_scores, k)
                ndcg_before[k].append(ndcg_before_k)
                ndcg_after[k].append(ndcg_after_k)
            
            # MRR
            mrr_before = self._calculate_mrr(initial_results, relevant_docs)
            mrr_after = self._calculate_mrr(reranked_results, relevant_docs)
            mrr_before_list.append(mrr_before)
            mrr_after_list.append(mrr_after)
            
            # Signal-to-Noise Ratio
            snr_before = self._calculate_snr(initial_results[:10], relevant_docs)
            snr_after = self._calculate_snr(reranked_results[:10], relevant_docs)
            snr_before_list.append(snr_before)
            snr_after_list.append(snr_after)
        
        # 计算改进度
        precision_gain_at_k = {}
        ndcg_improvement_at_k = {}
        
        for k in k_values:
            p_gain = np.mean(precision_after[k]) - np.mean(precision_before[k])
            precision_gain_at_k[k] = p_gain
            
            ndcg_imp = np.mean(ndcg_after[k]) - np.mean(ndcg_before[k])
            ndcg_improvement_at_k[k] = ndcg_imp
        
        mrr_improvement = np.mean(mrr_after_list) - np.mean(mrr_before_list)
        snr_improvement = np.mean(snr_after_list) - np.mean(snr_before_list)
        avg_latency = np.mean(latencies)
        
        # 延迟-质量比 (质量提升 / 延迟成本)
        # 质量提升用 NDCG@3 improvement 表示
        quality_gain = ndcg_improvement_at_k.get(3, 0.0)
        latency_cost = avg_latency / 1000  # 转换为秒
        latency_quality_ratio = quality_gain / latency_cost if latency_cost > 0 else 0.0
        
        metrics = RerankingMetrics(
            precision_gain_at_k=precision_gain_at_k,
            ndcg_improvement_at_k=ndcg_improvement_at_k,
            mrr_improvement=mrr_improvement,
            signal_to_noise_improvement=snr_improvement,
            avg_latency_ms=avg_latency,
            latency_quality_ratio=latency_quality_ratio,
            total_queries=len(test_queries)
        )
        
        self._display_results(metrics)
        return metrics
    
    def _calculate_precision(self, retrieved: List[str], relevant: set) -> float:
        """计算 Precision"""
        if not retrieved:
            return 0.0
        relevant_count = sum(1 for doc in retrieved if doc in relevant)
        return relevant_count / len(retrieved)
    
    def _calculate_ndcg(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        计算 NDCG@K
        
        Reference: Järvelin & Kekäläinen (2002)
        "Cumulative gain-based evaluation of IR techniques"
        """
        if not retrieved:
            return 0.0
        
        # DCG
        dcg = 0.0
        for idx, doc in enumerate(retrieved[:k], 1):
            relevance = relevance_scores.get(doc, 0.0)
            dcg += relevance / np.log2(idx + 1)
        
        # IDCG
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, retrieved: List[str], relevant: set) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)
        
        Reference: Classical IR metric
        """
        for idx, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / idx
        return 0.0
    
    def _calculate_snr(self, retrieved: List[str], relevant: set) -> float:
        """
        计算信噪比 (Signal-to-Noise Ratio)
        
        Signal: 相关文档数
        Noise: 不相关文档数
        SNR = Signal / (Signal + Noise + epsilon)
        """
        if not retrieved:
            return 0.0
        
        signal = sum(1 for doc in retrieved if doc in relevant)
        noise = len(retrieved) - signal
        epsilon = 1e-6  # 避免除零
        
        return signal / (signal + noise + epsilon)
    
    def _display_results(self, metrics: RerankingMetrics):
        """显示评估结果"""
        print(f"\n{'='*70}")
        print("📈 重排质量指标 (Academic Metrics)")
        print(f"{'='*70}")
        
        print(f"\n📊 Precision Gain @K:")
        print(f"  理论依据: Learning to Rank (Liu 2009)")
        for k in sorted(metrics.precision_gain_at_k.keys()):
            gain = metrics.precision_gain_at_k[k]
            sign = "+" if gain >= 0 else ""
            print(f"  • P@{k:2d}: {sign}{gain:.2%}")
        
        print(f"\n📊 NDCG Improvement @K:")
        print(f"  理论依据: Järvelin & Kekäläinen (2002)")
        for k in sorted(metrics.ndcg_improvement_at_k.keys()):
            imp = metrics.ndcg_improvement_at_k[k]
            sign = "+" if imp >= 0 else ""
            print(f"  • NDCG@{k:2d}: {sign}{imp:.2%}")
        
        print(f"\n📊 其他指标:")
        mrr_sign = "+" if metrics.mrr_improvement >= 0 else ""
        snr_sign = "+" if metrics.signal_to_noise_improvement >= 0 else ""
        print(f"  • MRR Improvement:    {mrr_sign}{metrics.mrr_improvement:.4f}")
        print(f"  • SNR Improvement:    {snr_sign}{metrics.signal_to_noise_improvement:.4f}")
        print(f"  • Avg Latency:        {metrics.avg_latency_ms:.2f}ms")
        print(f"  • Quality/Latency:    {metrics.latency_quality_ratio:.4f}")
        
        print(f"\n🎯 总体评分: {metrics._overall_score():.2%}")
        print(f"  权重: NDCG_Imp(40%) + Precision_Gain(30%) + MRR_Imp(20%) + SNR_Imp(10%)")
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
}


async def mock_initial_retrieval(query: str, top_k: int) -> List[str]:
    """模拟初始检索 (简单的关键词匹配)"""
    query_words = set(query.lower().split())
    scores = {}
    for doc_id, doc_text in MOCK_DOCS.items():
        doc_words = set(doc_text.lower().split())
        overlap = len(query_words & doc_words)
        scores[doc_id] = overlap
    
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs[:top_k]]


async def mock_reranking(query: str, doc_ids: List[str]) -> List[str]:
    """模拟重排 (基于更复杂的相似度计算)"""
    # 简化：添加一些延迟，模拟重排计算
    await asyncio.sleep(0.01)  # 10ms 延迟
    
    # 重排：根据文档与查询的更精细匹配
    query_words = query.lower().split()
    scores = {}
    for doc_id in doc_ids:
        doc_text = MOCK_DOCS[doc_id].lower()
        # 更复杂的评分：考虑词序、位置等
        score = sum(
            doc_text.index(word) if word in doc_text else 0
            for word in query_words
        )
        scores[doc_id] = score
    
    # 降序排列 (分数低的在前，因为是索引)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1])
    return [doc_id for doc_id, score in sorted_docs]


async def test_reranking_evaluator():
    """测试重排评估器"""
    
    evaluator = RerankingEvaluator(
        initial_retrieval_func=mock_initial_retrieval,
        reranking_func=mock_reranking
    )
    
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
            "relevant_docs": ["doc5"],
            "relevance_scores": {"doc5": 1.0}
        },
    ]
    
    # 运行评估
    metrics = await evaluator.evaluate(
        test_queries=test_queries,
        k_values=[1, 3, 5]
    )
    
    print(f"\n完整评估结果:\n{metrics.to_dict()}")


if __name__ == "__main__":
    asyncio.run(test_reranking_evaluator())
