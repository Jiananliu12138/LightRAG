#!/usr/bin/env python3
"""
快速入门示例：评估 RAG 系统的分块和检索性能

这个脚本展示了如何使用评估系统来评估和对比不同的配置。
"""

import asyncio
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from component_evaluators import (
    ChunkingEvaluator,
    RetrievalEvaluator
)


async def quick_start():
    """快速入门示例"""
    
    print(f"\n{'='*80}")
    print("🚀 RAG 评估系统 - 快速入门示例")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # 示例 1: 评估分块质量
    # ========================================================================
    
    print("📚 示例 1: 评估分块质量\n")
    
    # 准备测试文档
    sample_doc = """
    LightRAG is a Simple and Fast Retrieval-Augmented Generation framework. 
    The framework was developed by HKUDS (Hong Kong University Data Science Lab). 
    LightRAG provides developers with tools to build RAG applications efficiently.
    
    Large language models face several limitations. LLMs have a knowledge cutoff 
    date that prevents them from accessing recent information. Large language 
    models generate hallucinations when providing responses without factual grounding. 
    LLMs lack domain-specific expertise in specialized fields.
    
    LightRAG solves these problems by combining large language models with external 
    knowledge retrieval. The framework ensures accurate responses by grounding LLM 
    outputs in actual documents. LightRAG provides contextual responses that reduce 
    hallucinations significantly. The system enables efficient retrieval from external 
    knowledge bases to supplement LLM capabilities.
    """.strip()
    
    # 方法 1: 按固定大小分块 (256 字符, 50 字符重叠)
    def chunk_fixed_size(text, size=256, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start += (size - overlap)
        return chunks
    
    # 方法 2: 按段落分块
    def chunk_by_paragraph(text):
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # 方法 3: 按句子分块 (每个 chunk 最多 3 句)
    def chunk_by_sentence(text, max_sentences=3):
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = '. '.join(sentences[i:i+max_sentences]) + '.'
            chunks.append(chunk)
        return chunks
    
    # 创建评估器
    chunking_evaluator = ChunkingEvaluator()
    
    # 评估三种分块方法
    print("方法 1: 固定大小分块 (256 字符)")
    chunks1 = chunk_fixed_size(sample_doc, size=256, overlap=50)
    metrics1 = await chunking_evaluator.evaluate(sample_doc, chunks1)
    
    print("\n方法 2: 按段落分块")
    chunks2 = chunk_by_paragraph(sample_doc)
    metrics2 = await chunking_evaluator.evaluate(sample_doc, chunks2)
    
    print("\n方法 3: 按句子分块 (每 3 句)")
    chunks3 = chunk_by_sentence(sample_doc, max_sentences=3)
    metrics3 = await chunking_evaluator.evaluate(sample_doc, chunks3)
    
    # 对比结果
    print(f"\n{'='*80}")
    print("📊 分块方法对比")
    print(f"{'='*80}")
    print(f"固定大小分块: {metrics1._overall_score():.2%}")
    print(f"按段落分块:   {metrics2._overall_score():.2%}")
    print(f"按句子分块:   {metrics3._overall_score():.2%}")
    
    # 找出最佳方法
    best_score = max(metrics1._overall_score(), metrics2._overall_score(), metrics3._overall_score())
    if best_score == metrics1._overall_score():
        print("\n🏆 最佳方法: 固定大小分块")
    elif best_score == metrics2._overall_score():
        print("\n🏆 最佳方法: 按段落分块")
    else:
        print("\n🏆 最佳方法: 按句子分块")
    
    # ========================================================================
    # 示例 2: 评估检索质量
    # ========================================================================
    
    print(f"\n\n{'='*80}")
    print("📚 示例 2: 评估检索质量")
    print(f"{'='*80}\n")
    
    # 模拟文档库
    mock_docs = {
        "doc1": "LightRAG is a Simple and Fast Retrieval-Augmented Generation framework",
        "doc2": "LightRAG was developed by HKUDS",
        "doc3": "Python is a high-level programming language",
        "doc4": "RAG systems combine retrieval and generation",
        "doc5": "Vector databases store embeddings efficiently",
        "doc6": "Knowledge graphs represent structured information",
        "doc7": "RAGAS is a framework for evaluating RAG systems",
    }
    
    # 模拟检索函数
    async def mock_retrieval(query: str, top_k: int):
        """简单的关键词匹配检索"""
        query_words = set(query.lower().split())
        scores = {}
        for doc_id, doc_text in mock_docs.items():
            doc_words = set(doc_text.lower().split())
            overlap = len(query_words & doc_words)
            scores[doc_id] = overlap
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs[:top_k]]
    
    # 创建测试查询
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
    
    # 创建评估器
    retrieval_evaluator = RetrievalEvaluator(retrieval_func=mock_retrieval)
    
    # 评估检索质量
    print("评估检索性能...\n")
    retrieval_metrics = await retrieval_evaluator.evaluate(
        test_queries=test_queries,
        k_values=[1, 3, 5]
    )
    
    # ========================================================================
    # 总结
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("✅ 快速入门示例完成！")
    print(f"{'='*80}\n")
    
    print("📌 关键要点:")
    print("  1. 分块评估器可以帮助您选择最佳的分块策略")
    print("  2. 检索评估器可以衡量不同检索方法的性能")
    print("  3. 使用评估系统可以系统化地优化您的 RAG 系统")
    
    print(f"\n📚 下一步:")
    print("  • 阅读完整指南: RAG_EVALUATION_GUIDE.md")
    print("  • 运行完整评估: python evaluate_lightrag_complete.py")
    print("  • 自定义评估数据: 修改测试数据以适配您的场景")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(quick_start())
