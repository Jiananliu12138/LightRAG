#!/usr/bin/env python3
"""
嵌入质量评估器 (Embedding Evaluator)

评估指标:
1. 语义相似度保持 (Semantic Similarity Preservation)
2. 主题区分度 (Topic Separation)
3. 检索准确率 (Retrieval Accuracy)
4. 降维质量 (Dimensionality Quality)
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class EmbeddingMetrics:
    """嵌入评估指标"""
    semantic_similarity_preservation: float  # 语义相似度保持 (0-1)
    topic_separation: float                  # 主题区分度 (0-1)
    retrieval_accuracy: float                # 检索准确率 (0-1)
    intra_cluster_similarity: float          # 簇内相似度 (0-1)
    inter_cluster_distance: float            # 簇间距离 (0-1)
    dimension: int                           # 嵌入维度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_similarity_preservation": round(self.semantic_similarity_preservation, 4),
            "topic_separation": round(self.topic_separation, 4),
            "retrieval_accuracy": round(self.retrieval_accuracy, 4),
            "intra_cluster_similarity": round(self.intra_cluster_similarity, 4),
            "inter_cluster_distance": round(self.inter_cluster_distance, 4),
            "dimension": self.dimension,
            "overall_score": round(self._overall_score(), 4)
        }
    
    def _overall_score(self) -> float:
        """计算总体分数"""
        return (
            self.semantic_similarity_preservation * 0.3 +
            self.topic_separation * 0.25 +
            self.retrieval_accuracy * 0.35 +
            self.intra_cluster_similarity * 0.05 +
            self.inter_cluster_distance * 0.05
        )


class EmbeddingEvaluator:
    """嵌入质量评估器"""
    
    def __init__(self, embedding_func: Callable):
        """
        Args:
            embedding_func: 嵌入函数，输入文本列表，返回嵌入向量数组
                           示例: async def embed(texts: List[str]) -> np.ndarray
        """
        self.embedding_func = embedding_func
    
    async def evaluate(
        self,
        test_pairs: List[Tuple[str, str, float]],  # (text1, text2, 人工相似度分数)
        test_clusters: List[List[str]] = None,     # 不同主题的文本簇
        retrieval_test: List[Tuple[str, List[str], int]] = None  # (query, candidates, 正确答案索引)
    ) -> EmbeddingMetrics:
        """
        评估嵌入质量
        
        Args:
            test_pairs: 文本对及其人工标注的相似度 (0-1)
            test_clusters: 不同主题的文本簇，用于评估主题区分
            retrieval_test: 检索测试用例 (query, 候选文本列表, 正确答案索引)
        
        Returns:
            EmbeddingMetrics: 评估指标
        """
        print(f"\n{'='*70}")
        print("🧬 嵌入质量评估")
        print(f"{'='*70}")
        
        # 1. 语义相似度保持
        similarity_preservation = await self._evaluate_similarity_preservation(test_pairs)
        
        # 2. 主题区分度
        if test_clusters:
            topic_separation, intra_sim, inter_dist = await self._evaluate_topic_separation(test_clusters)
        else:
            topic_separation, intra_sim, inter_dist = 0.0, 0.0, 0.0
        
        # 3. 检索准确率
        if retrieval_test:
            retrieval_accuracy = await self._evaluate_retrieval_accuracy(retrieval_test)
        else:
            retrieval_accuracy = 0.0
        
        # 获取嵌入维度
        sample_embedding = await self.embedding_func([test_pairs[0][0]])
        dimension = sample_embedding.shape[1] if len(sample_embedding.shape) > 1 else len(sample_embedding)
        
        metrics = EmbeddingMetrics(
            semantic_similarity_preservation=similarity_preservation,
            topic_separation=topic_separation,
            retrieval_accuracy=retrieval_accuracy,
            intra_cluster_similarity=intra_sim,
            inter_cluster_distance=inter_dist,
            dimension=dimension
        )
        
        self._display_results(metrics)
        return metrics
    
    async def _evaluate_similarity_preservation(
        self,
        test_pairs: List[Tuple[str, str, float]]
    ) -> float:
        """
        评估语义相似度保持
        
        方法：计算嵌入相似度与人工标注相似度的相关性 (Pearson/Spearman)
        """
        if not test_pairs:
            return 0.0
        
        texts1, texts2, human_scores = [], [], []
        for text1, text2, score in test_pairs:
            texts1.append(text1)
            texts2.append(text2)
            human_scores.append(score)
        
        # 获取嵌入
        embeddings1 = await self.embedding_func(texts1)
        embeddings2 = await self.embedding_func(texts2)
        
        # 计算嵌入相似度
        embedding_scores = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            embedding_scores.append(similarity)
        
        # 计算相关性 (Pearson)
        correlation = np.corrcoef(human_scores, embedding_scores)[0, 1]
        
        # 转换为 0-1 分数（相关性 -1 到 1，转换为 0 到 1）
        score = (correlation + 1) / 2
        
        return score
    
    async def _evaluate_topic_separation(
        self,
        test_clusters: List[List[str]]
    ) -> Tuple[float, float, float]:
        """
        评估主题区分度
        
        方法：
        - 簇内相似度应该高
        - 簇间距离应该大
        
        Returns:
            (主题区分度, 簇内相似度, 簇间距离)
        """
        if len(test_clusters) < 2:
            return 0.0, 0.0, 0.0
        
        # 获取所有文本的嵌入
        all_texts = []
        cluster_labels = []
        for cluster_id, cluster in enumerate(test_clusters):
            all_texts.extend(cluster)
            cluster_labels.extend([cluster_id] * len(cluster))
        
        embeddings = await self.embedding_func(all_texts)
        
        # 计算簇内相似度 (Intra-cluster similarity)
        intra_similarities = []
        for cluster_id, cluster in enumerate(test_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_indices]
            
            if len(cluster_embeddings) > 1:
                # 计算簇内所有对的平均相似度
                sim_matrix = cosine_similarity(cluster_embeddings)
                # 去除对角线（自己与自己的相似度）
                mask = ~np.eye(len(cluster_embeddings), dtype=bool)
                intra_sim = sim_matrix[mask].mean()
                intra_similarities.append(intra_sim)
        
        avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0.0
        
        # 计算簇间距离 (Inter-cluster distance)
        cluster_centroids = []
        for cluster_id in range(len(test_clusters)):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = cluster_embeddings.mean(axis=0)
            cluster_centroids.append(centroid)
        
        # 计算所有簇中心对的距离
        inter_distances = []
        for i in range(len(cluster_centroids)):
            for j in range(i + 1, len(cluster_centroids)):
                distance = 1 - cosine_similarity([cluster_centroids[i]], [cluster_centroids[j]])[0][0]
                inter_distances.append(distance)
        
        avg_inter_distance = np.mean(inter_distances) if inter_distances else 0.0
        
        # 主题区分度 = 簇间距离 - (1 - 簇内相似度)
        # 簇间距离大且簇内相似度高 = 区分度好
        topic_separation = (avg_inter_distance + avg_intra_similarity) / 2
        
        return topic_separation, avg_intra_similarity, avg_inter_distance
    
    async def _evaluate_retrieval_accuracy(
        self,
        retrieval_test: List[Tuple[str, List[str], int]]
    ) -> float:
        """
        评估检索准确率
        
        方法：
        - 对于每个查询，检索最相似的候选文本
        - 计算 Top-1 准确率
        """
        if not retrieval_test:
            return 0.0
        
        correct_count = 0
        
        for query, candidates, correct_idx in retrieval_test:
            # 获取查询和候选文本的嵌入
            query_embedding = await self.embedding_func([query])
            candidate_embeddings = await self.embedding_func(candidates)
            
            # 计算相似度
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # 找到最相似的候选
            predicted_idx = np.argmax(similarities)
            
            if predicted_idx == correct_idx:
                correct_count += 1
        
        accuracy = correct_count / len(retrieval_test)
        return accuracy
    
    def _display_results(self, metrics: EmbeddingMetrics):
        """显示评估结果"""
        print(f"\n📈 嵌入质量指标:")
        print(f"  • 语义相似度保持: {metrics.semantic_similarity_preservation:.2%}")
        print(f"  • 主题区分度:     {metrics.topic_separation:.2%}")
        print(f"  • 检索准确率:     {metrics.retrieval_accuracy:.2%}")
        print(f"  • 簇内相似度:     {metrics.intra_cluster_similarity:.2%}")
        print(f"  • 簇间距离:       {metrics.inter_cluster_distance:.2%}")
        print(f"\n📊 嵌入信息:")
        print(f"  • 嵌入维度: {metrics.dimension}")
        print(f"\n🎯 总体评分: {metrics._overall_score():.2%}")
        print(f"{'='*70}\n")


# ============================================================================
# 示例使用
# ============================================================================

async def mock_embedding_func(texts: List[str]) -> np.ndarray:
    """模拟嵌入函数（用于测试）"""
    # 简单的词袋模型嵌入
    embeddings = []
    for text in texts:
        # 简化：使用文本长度和单词数作为特征
        words = text.lower().split()
        embedding = np.random.randn(128)  # 128维随机向量
        # 添加一些基于文本内容的特征
        embedding[0] = len(text) / 100
        embedding[1] = len(words) / 10
        embeddings.append(embedding)
    return np.array(embeddings)


async def test_embedding_evaluator():
    """测试嵌入评估器"""
    
    evaluator = EmbeddingEvaluator(embedding_func=mock_embedding_func)
    
    # 1. 语义相似度测试对
    test_pairs = [
        ("LightRAG is a RAG framework", "LightRAG is a retrieval system", 0.9),
        ("LightRAG is a RAG framework", "Python is a programming language", 0.1),
        ("The cat sat on the mat", "A cat was sitting on a mat", 0.95),
        ("The cat sat on the mat", "Dogs are loyal animals", 0.2),
    ]
    
    # 2. 主题簇测试
    test_clusters = [
        # RAG 主题
        [
            "LightRAG is a retrieval-augmented generation framework",
            "RAG systems combine retrieval and generation",
            "Retrieval-augmented generation improves LLM accuracy"
        ],
        # Python 主题
        [
            "Python is a high-level programming language",
            "Python supports object-oriented programming",
            "Python has a rich ecosystem of libraries"
        ],
        # 数据库主题
        [
            "MySQL is a relational database",
            "PostgreSQL supports advanced SQL features",
            "Databases store and manage structured data"
        ]
    ]
    
    # 3. 检索测试
    retrieval_test = [
        (
            "What is LightRAG?",
            [
                "LightRAG is a RAG framework for AI applications",
                "Python is a programming language",
                "Databases store data efficiently"
            ],
            0  # 正确答案索引
        ),
        (
            "Tell me about Python",
            [
                "LightRAG is a RAG framework",
                "Python is a versatile programming language",
                "SQL is used for database queries"
            ],
            1  # 正确答案索引
        ),
    ]
    
    # 运行评估
    metrics = await evaluator.evaluate(
        test_pairs=test_pairs,
        test_clusters=test_clusters,
        retrieval_test=retrieval_test
    )
    
    print(f"评估结果: {metrics.to_dict()}")


if __name__ == "__main__":
    asyncio.run(test_embedding_evaluator())
