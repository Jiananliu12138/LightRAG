#!/usr/bin/env python3
"""
LightRAG 完整评估脚本

功能:
1. 评估 LightRAG 的所有组件（分块、嵌入、检索、生成）
2. 支持多种配置对比
3. 生成详细的评估报告
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import aiohttp
import numpy as np
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from rag_evaluator_system import (
    RAGEvaluationSystem,
    RAGSystemConfig,
    RAGEvaluationResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightRAGEvaluator:
    """LightRAG 评估器"""
    
    def __init__(
        self,
        lightrag_api_url: str = "http://localhost:9621",
        ollama_api_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "qwen2.5:7b-instruct"
    ):
        """
        Args:
            lightrag_api_url: LightRAG API 地址
            ollama_api_url: Ollama API 地址
            embedding_model: 嵌入模型名称
            llm_model: LLM 模型名称
        """
        self.lightrag_api_url = lightrag_api_url
        self.ollama_api_url = ollama_api_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
    
    async def get_embedding_func(self):
        """创建嵌入函数"""
        async def embed_texts(texts: List[str]) -> np.ndarray:
            """使用 Ollama 嵌入文本"""
            embeddings = []
            async with aiohttp.ClientSession() as session:
                for text in texts:
                    async with session.post(
                        f"{self.ollama_api_url}/api/embeddings",
                        json={
                            "model": self.embedding_model,
                            "prompt": text
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            embeddings.append(result["embedding"])
                        else:
                            logger.error(f"嵌入请求失败: {response.status}")
                            embeddings.append([0.0] * 768)  # 回退
            return np.array(embeddings)
        
        return embed_texts
    
    async def get_retrieval_func(self, mode: str = "hybrid"):
        """创建检索函数"""
        async def retrieve_docs(query: str, top_k: int) -> List[str]:
            """使用 LightRAG 检索文档"""
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.lightrag_api_url}/query",
                    json={
                        "query": query,
                        "mode": mode,
                        "only_need_context": True,
                        "top_k": top_k
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # 提取检索到的文档 ID
                        # 注意：需要根据实际 API 响应调整
                        contexts = result.get("contexts", [])
                        return [ctx.get("chunk_id", f"doc_{i}") for i, ctx in enumerate(contexts[:top_k])]
                    else:
                        logger.error(f"检索请求失败: {response.status}")
                        return []
        
        return retrieve_docs
    
    async def get_end_to_end_func(self):
        """创建端到端评估函数（调用现有的 RAGAS 评估）"""
        async def evaluate_e2e(test_cases: List) -> Dict[str, Any]:
            """
            运行 RAGAS 评估
            
            注意：这里应该调用 eval_rag_quality.py 中的评估逻辑
            为了演示，这里返回模拟数据
            """
            # TODO: 集成实际的 RAGAS 评估
            # 可以通过导入 eval_rag_quality.py 中的 RAGQualityEvaluator 实现
            
            logger.info("运行端到端评估（RAGAS）...")
            
            # 模拟评估结果
            return {
                "faithfulness": 0.85,
                "answer_relevancy": 0.78,
                "context_recall": 0.92,
                "context_precision": 0.88,
                "ragas_score": 0.86
            }
        
        return evaluate_e2e
    
    def load_sample_documents(self, docs_dir: Path = None) -> List[str]:
        """加载示例文档"""
        if docs_dir is None:
            docs_dir = Path(__file__).parent / "sample_documents"
        
        if not docs_dir.exists():
            logger.warning(f"文档目录不存在: {docs_dir}")
            return []
        
        documents = []
        for file_path in docs_dir.glob("*.md"):
            try:
                content = file_path.read_text(encoding='utf-8')
                documents.append(content)
            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {e}")
        
        logger.info(f"加载了 {len(documents)} 个文档")
        return documents
    
    def create_test_chunks(self, document: str, chunk_size: int, overlap: int) -> List[str]:
        """创建测试分块"""
        chunks = []
        start = 0
        doc_len = len(document)
        
        while start < doc_len:
            end = min(start + chunk_size, doc_len)
            chunk = document[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        
        return chunks
    
    def create_embedding_test_data(self) -> tuple:
        """创建嵌入评估测试数据"""
        # 语义相似度测试对
        test_pairs = [
            ("LightRAG is a RAG framework", "LightRAG is a retrieval system", 0.9),
            ("LightRAG is a RAG framework", "Python is a programming language", 0.1),
            ("RAG systems combine retrieval and generation", "Retrieval-augmented generation merges search and LLMs", 0.95),
            ("Knowledge graphs represent structured data", "Unstructured text contains free-form information", 0.2),
            ("Vector databases store embeddings", "Embedding databases use vector search", 0.9),
        ]
        
        # 主题簇测试
        test_clusters = [
            # RAG 主题
            [
                "LightRAG is a retrieval-augmented generation framework",
                "RAG systems combine retrieval and generation",
                "Retrieval-augmented generation improves LLM accuracy"
            ],
            # 数据库主题
            [
                "Vector databases store embeddings efficiently",
                "Graph databases represent relationships",
                "NoSQL databases handle unstructured data"
            ],
            # AI 模型主题
            [
                "Large language models generate human-like text",
                "Embedding models convert text to vectors",
                "Neural networks learn from data"
            ]
        ]
        
        # 检索测试
        retrieval_test = [
            (
                "What is LightRAG?",
                [
                    "LightRAG is a RAG framework",
                    "Python is a language",
                    "Databases store data"
                ],
                0
            ),
            (
                "How do vector databases work?",
                [
                    "RAG systems use retrieval",
                    "Vector databases store embeddings",
                    "LLMs generate text"
                ],
                1
            ),
        ]
        
        return test_pairs, test_clusters, retrieval_test
    
    def create_retrieval_test_data(self) -> List[Dict[str, Any]]:
        """创建检索评估测试数据"""
        # 注意：这里的 doc ID 需要与实际插入 LightRAG 的文档对应
        return [
            {
                "query": "What is LightRAG?",
                "relevant_docs": ["doc1", "doc2"],
                "relevance_scores": {"doc1": 1.0, "doc2": 0.8}
            },
            {
                "query": "How does RAG work?",
                "relevant_docs": ["doc1", "doc3"],
                "relevance_scores": {"doc1": 0.9, "doc3": 1.0}
            },
            {
                "query": "What databases does LightRAG support?",
                "relevant_docs": ["doc4"],
                "relevance_scores": {"doc4": 1.0}
            },
        ]


async def main():
    """主评估流程"""
    
    print(f"\n{'='*80}")
    print("🚀 LightRAG 完整评估系统")
    print(f"{'='*80}\n")
    
    # 初始化评估系统
    eval_system = RAGEvaluationSystem(
        output_dir=Path("./lightrag_evaluation_results")
    )
    
    lightrag_eval = LightRAGEvaluator()
    
    # ========================================================================
    # 配置1: 小 chunk 配置
    # ========================================================================
    config1 = RAGSystemConfig(
        name="LightRAG_Small_Chunks_256",
        chunking_method="fixed_size",
        chunk_size=256,
        chunk_overlap=50,
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        retrieval_method="hybrid",
        top_k=10,
        llm_model="qwen2.5:7b-instruct"
    )
    
    # ========================================================================
    # 配置2: 大 chunk 配置
    # ========================================================================
    config2 = RAGSystemConfig(
        name="LightRAG_Large_Chunks_512",
        chunking_method="fixed_size",
        chunk_size=512,
        chunk_overlap=100,
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        retrieval_method="hybrid",
        top_k=10,
        llm_model="qwen2.5:7b-instruct"
    )
    
    # ========================================================================
    # 配置3: 语义分块（如果支持）
    # ========================================================================
    config3 = RAGSystemConfig(
        name="LightRAG_Semantic_Chunks",
        chunking_method="semantic",
        chunk_size=400,
        chunk_overlap=80,
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        retrieval_method="hybrid",
        top_k=10,
        llm_model="qwen2.5:7b-instruct"
    )
    
    # 加载测试文档
    docs = lightrag_eval.load_sample_documents()
    if not docs:
        logger.error("❌ 未找到测试文档，请确保 sample_documents 目录存在")
        return
    
    test_doc = "\n\n".join(docs)  # 合并所有文档
    
    # 创建测试数据
    embedding_test_pairs, embedding_test_clusters, embedding_retrieval_test = \
        lightrag_eval.create_embedding_test_data()
    
    retrieval_test_queries = lightrag_eval.create_retrieval_test_data()
    
    # 获取评估函数
    embedding_func = await lightrag_eval.get_embedding_func()
    retrieval_func = await lightrag_eval.get_retrieval_func(mode="hybrid")
    end_to_end_func = await lightrag_eval.get_end_to_end_func()
    
    # ========================================================================
    # 评估配置1
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("评估配置 1: 小 Chunk (256)")
    logger.info("="*80)
    
    chunks1 = lightrag_eval.create_test_chunks(test_doc, 256, 50)
    
    result1 = await eval_system.evaluate_system(
        config=config1,
        test_document=test_doc,
        chunks=chunks1,
        embedding_func=embedding_func,
        embedding_test_pairs=embedding_test_pairs,
        embedding_test_clusters=embedding_test_clusters,
        retrieval_func=retrieval_func,
        retrieval_test_queries=retrieval_test_queries,
        end_to_end_func=end_to_end_func,
        end_to_end_test_cases=[],  # 传入实际的测试用例
        evaluate_chunking=True,
        evaluate_embedding=True,
        evaluate_retrieval=True,
        evaluate_end_to_end=False  # 可选：启用端到端评估
    )
    
    # ========================================================================
    # 评估配置2
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("评估配置 2: 大 Chunk (512)")
    logger.info("="*80)
    
    chunks2 = lightrag_eval.create_test_chunks(test_doc, 512, 100)
    
    result2 = await eval_system.evaluate_system(
        config=config2,
        test_document=test_doc,
        chunks=chunks2,
        embedding_func=embedding_func,
        embedding_test_pairs=embedding_test_pairs,
        embedding_test_clusters=embedding_test_clusters,
        retrieval_func=retrieval_func,
        retrieval_test_queries=retrieval_test_queries,
        end_to_end_func=end_to_end_func,
        end_to_end_test_cases=[],
        evaluate_chunking=True,
        evaluate_embedding=True,
        evaluate_retrieval=True,
        evaluate_end_to_end=False
    )
    
    # ========================================================================
    # 对比分析
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("生成对比分析")
    logger.info("="*80)
    
    comparison_df = eval_system.compare_systems()
    
    # ========================================================================
    # 生成报告
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("生成评估报告")
    logger.info("="*80)
    
    eval_system.generate_report()
    
    print(f"\n{'='*80}")
    print("✅ 评估完成！")
    print(f"{'='*80}\n")
    print(f"📁 结果保存在: {eval_system.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
