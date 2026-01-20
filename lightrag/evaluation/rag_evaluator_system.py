#!/usr/bin/env python3
"""
完整的 RAG 评估系统 (RAG Evaluation System)

功能:
1. 组件级评估：分块、嵌入、检索、生成
2. 端到端评估：使用 RAGAS
3. 系统对比：不同配置/系统的性能对比
4. 报告生成：生成详细的评估报告
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pandas as pd

from component_evaluators import (
    ChunkingEvaluator, ChunkingMetrics,
    EmbeddingEvaluator, EmbeddingMetrics,
    RetrievalEvaluator, RetrievalMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGSystemConfig:
    """RAG 系统配置"""
    name: str                          # 系统名称
    chunking_method: str               # 分块方法 (e.g., "fixed_size", "sentence", "semantic")
    chunk_size: int                    # Chunk 大小
    chunk_overlap: int                 # Chunk 重叠
    embedding_model: str               # 嵌入模型
    embedding_dim: int                 # 嵌入维度
    retrieval_method: str              # 检索方法 (e.g., "vector", "graph", "hybrid")
    top_k: int                         # 检索 Top-K
    llm_model: str                     # 生成模型
    rerank_model: Optional[str] = None # 重排模型
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGEvaluationResult:
    """RAG 评估结果"""
    config: RAGSystemConfig
    chunking_metrics: Optional[ChunkingMetrics]
    embedding_metrics: Optional[EmbeddingMetrics]
    retrieval_metrics: Optional[RetrievalMetrics]
    end_to_end_metrics: Optional[Dict[str, Any]]  # RAGAS 指标
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "chunking_metrics": self.chunking_metrics.to_dict() if self.chunking_metrics else None,
            "embedding_metrics": self.embedding_metrics.to_dict() if self.embedding_metrics else None,
            "retrieval_metrics": self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            "end_to_end_metrics": self.end_to_end_metrics,
            "timestamp": self.timestamp
        }


class RAGEvaluationSystem:
    """完整的 RAG 评估系统"""
    
    def __init__(
        self,
        output_dir: Path = Path("./evaluation_results")
    ):
        """
        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[RAGEvaluationResult] = []
    
    async def evaluate_system(
        self,
        config: RAGSystemConfig,
        # 分块评估参数
        test_document: Optional[str] = None,
        chunks: Optional[List[str]] = None,
        # 嵌入评估参数
        embedding_func: Optional[Callable] = None,
        embedding_test_pairs: Optional[List] = None,
        embedding_test_clusters: Optional[List] = None,
        # 检索评估参数
        retrieval_func: Optional[Callable] = None,
        retrieval_test_queries: Optional[List] = None,
        # 端到端评估参数
        end_to_end_func: Optional[Callable] = None,
        end_to_end_test_cases: Optional[List] = None,
        # 评估选项
        evaluate_chunking: bool = True,
        evaluate_embedding: bool = True,
        evaluate_retrieval: bool = True,
        evaluate_end_to_end: bool = True
    ) -> RAGEvaluationResult:
        """
        评估完整的 RAG 系统
        
        Args:
            config: RAG 系统配置
            test_document: 用于分块评估的测试文档
            chunks: 分块后的结果
            embedding_func: 嵌入函数
            embedding_test_pairs: 嵌入评估测试对
            embedding_test_clusters: 嵌入评估测试簇
            retrieval_func: 检索函数
            retrieval_test_queries: 检索评估测试查询
            end_to_end_func: 端到端评估函数（RAGAS）
            end_to_end_test_cases: 端到端评估测试用例
            evaluate_chunking: 是否评估分块
            evaluate_embedding: 是否评估嵌入
            evaluate_retrieval: 是否评估检索
            evaluate_end_to_end: 是否评估端到端
        
        Returns:
            RAGEvaluationResult: 评估结果
        """
        print(f"\n{'='*80}")
        print(f"🚀 开始评估 RAG 系统: {config.name}")
        print(f"{'='*80}\n")
        
        chunking_metrics = None
        embedding_metrics = None
        retrieval_metrics = None
        end_to_end_metrics = None
        
        # 1. 分块评估
        if evaluate_chunking and test_document and chunks:
            logger.info("📊 步骤 1/4: 分块评估")
            chunking_evaluator = ChunkingEvaluator()
            chunking_metrics = await chunking_evaluator.evaluate(
                original_document=test_document,
                chunks=chunks
            )
        
        # 2. 嵌入评估
        if evaluate_embedding and embedding_func and embedding_test_pairs:
            logger.info("📊 步骤 2/4: 嵌入评估")
            embedding_evaluator = EmbeddingEvaluator(embedding_func=embedding_func)
            embedding_metrics = await embedding_evaluator.evaluate(
                test_pairs=embedding_test_pairs,
                test_clusters=embedding_test_clusters
            )
        
        # 3. 检索评估
        if evaluate_retrieval and retrieval_func and retrieval_test_queries:
            logger.info("📊 步骤 3/4: 检索评估")
            retrieval_evaluator = RetrievalEvaluator(retrieval_func=retrieval_func)
            retrieval_metrics = await retrieval_evaluator.evaluate(
                test_queries=retrieval_test_queries
            )
        
        # 4. 端到端评估 (RAGAS)
        if evaluate_end_to_end and end_to_end_func and end_to_end_test_cases:
            logger.info("📊 步骤 4/4: 端到端评估 (RAGAS)")
            end_to_end_metrics = await end_to_end_func(end_to_end_test_cases)
        
        # 创建评估结果
        result = RAGEvaluationResult(
            config=config,
            chunking_metrics=chunking_metrics,
            embedding_metrics=embedding_metrics,
            retrieval_metrics=retrieval_metrics,
            end_to_end_metrics=end_to_end_metrics,
            timestamp=datetime.now().isoformat()
        )
        
        # 保存结果
        self.results.append(result)
        self._save_result(result)
        
        print(f"\n{'='*80}")
        print(f"✅ 评估完成: {config.name}")
        print(f"{'='*80}\n")
        
        return result
    
    def compare_systems(
        self,
        results: Optional[List[RAGEvaluationResult]] = None
    ) -> pd.DataFrame:
        """
        对比多个 RAG 系统
        
        Args:
            results: 要对比的评估结果列表（默认使用所有已评估的系统）
        
        Returns:
            pd.DataFrame: 对比表格
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("没有可对比的结果")
            return pd.DataFrame()
        
        print(f"\n{'='*80}")
        print(f"📊 系统对比分析 ({len(results)} 个系统)")
        print(f"{'='*80}\n")
        
        # 构建对比表格
        comparison_data = []
        
        for result in results:
            row = {
                "系统名称": result.config.name,
                "分块方法": result.config.chunking_method,
                "Chunk大小": result.config.chunk_size,
                "嵌入模型": result.config.embedding_model,
                "嵌入维度": result.config.embedding_dim,
                "检索方法": result.config.retrieval_method,
                "Top-K": result.config.top_k,
                "LLM模型": result.config.llm_model,
            }
            
            # 添加分块指标
            if result.chunking_metrics:
                row["分块-总分"] = result.chunking_metrics._overall_score()
                row["分块-语义完整性"] = result.chunking_metrics.semantic_completeness
                row["分块-边界质量"] = result.chunking_metrics.boundary_quality
            
            # 添加嵌入指标
            if result.embedding_metrics:
                row["嵌入-总分"] = result.embedding_metrics._overall_score()
                row["嵌入-检索准确率"] = result.embedding_metrics.retrieval_accuracy
            
            # 添加检索指标
            if result.retrieval_metrics:
                row["检索-总分"] = result.retrieval_metrics._overall_score()
                row["检索-P@5"] = result.retrieval_metrics.precision_at_k.get(5, 0.0)
                row["检索-R@5"] = result.retrieval_metrics.recall_at_k.get(5, 0.0)
                row["检索-MRR"] = result.retrieval_metrics.mrr
            
            # 添加端到端指标 (RAGAS)
            if result.end_to_end_metrics:
                row["端到端-Faithfulness"] = result.end_to_end_metrics.get("faithfulness", 0.0)
                row["端到端-AnswerRelevancy"] = result.end_to_end_metrics.get("answer_relevancy", 0.0)
                row["端到端-ContextRecall"] = result.end_to_end_metrics.get("context_recall", 0.0)
                row["端到端-ContextPrecision"] = result.end_to_end_metrics.get("context_precision", 0.0)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 显示对比表格
        print(df.to_string(index=False))
        
        # 保存对比表格
        comparison_file = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 对比表格已保存: {comparison_file}")
        
        # 找出最佳系统
        self._highlight_best_systems(df)
        
        return df
    
    def _highlight_best_systems(self, df: pd.DataFrame):
        """高亮显示最佳系统"""
        print(f"\n{'='*80}")
        print("🏆 最佳系统")
        print(f"{'='*80}")
        
        metrics = [
            ("分块-总分", "分块质量最佳"),
            ("嵌入-总分", "嵌入质量最佳"),
            ("检索-总分", "检索质量最佳"),
            ("端到端-Faithfulness", "端到端-忠实度最佳"),
        ]
        
        for metric, description in metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_system = df.loc[best_idx, "系统名称"]
                best_score = df.loc[best_idx, metric]
                print(f"  • {description}: {best_system} ({best_score:.2%})")
    
    def _save_result(self, result: RAGEvaluationResult):
        """保存评估结果"""
        # 保存为 JSON
        result_file = self.output_dir / f"{result.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 评估结果已保存: {result_file}")
    
    def generate_report(self, output_file: Optional[Path] = None):
        """
        生成 HTML 评估报告
        
        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # TODO: 生成详细的 HTML 报告
        logger.info(f"📄 生成评估报告: {output_file}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG 系统评估报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🎯 RAG 系统评估报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>评估系统数: {len(self.results)}</p>
            
            <h2>评估结果摘要</h2>
            {self._generate_summary_html()}
            
            <h2>详细评估结果</h2>
            {self._generate_detailed_html()}
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ 报告已生成: {output_file}")
    
    def _generate_summary_html(self) -> str:
        """生成摘要 HTML"""
        # 简化版摘要
        return "<p>评估摘要（待实现）</p>"
    
    def _generate_detailed_html(self) -> str:
        """生成详细结果 HTML"""
        # 简化版详细结果
        return "<p>详细结果（待实现）</p>"


# ============================================================================
# 示例使用
# ============================================================================

async def mock_end_to_end_evaluation(test_cases: List) -> Dict[str, Any]:
    """模拟端到端评估（实际应该调用 RAGAS）"""
    return {
        "faithfulness": 0.85,
        "answer_relevancy": 0.78,
        "context_recall": 0.92,
        "context_precision": 0.88
    }


async def test_rag_evaluation_system():
    """测试 RAG 评估系统"""
    
    eval_system = RAGEvaluationSystem(output_dir=Path("./test_evaluation_results"))
    
    # 配置1: LightRAG with small chunks
    config1 = RAGSystemConfig(
        name="LightRAG_Small_Chunks",
        chunking_method="fixed_size",
        chunk_size=256,
        chunk_overlap=50,
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        retrieval_method="hybrid",
        top_k=10,
        llm_model="qwen2.5:7b-instruct"
    )
    
    # 配置2: LightRAG with large chunks
    config2 = RAGSystemConfig(
        name="LightRAG_Large_Chunks",
        chunking_method="fixed_size",
        chunk_size=512,
        chunk_overlap=100,
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        retrieval_method="hybrid",
        top_k=10,
        llm_model="qwen2.5:7b-instruct"
    )
    
    # 测试文档
    test_doc = "LightRAG is a Simple and Fast Retrieval-Augmented Generation framework. " * 10
    chunks1 = [test_doc[i:i+256] for i in range(0, len(test_doc), 256-50)]
    chunks2 = [test_doc[i:i+512] for i in range(0, len(test_doc), 512-100)]
    
    # 评估系统1
    result1 = await eval_system.evaluate_system(
        config=config1,
        test_document=test_doc,
        chunks=chunks1,
        evaluate_chunking=True,
        evaluate_embedding=False,  # 跳过嵌入评估（演示）
        evaluate_retrieval=False,  # 跳过检索评估（演示）
        evaluate_end_to_end=False  # 跳过端到端评估（演示）
    )
    
    # 评估系统2
    result2 = await eval_system.evaluate_system(
        config=config2,
        test_document=test_doc,
        chunks=chunks2,
        evaluate_chunking=True,
        evaluate_embedding=False,
        evaluate_retrieval=False,
        evaluate_end_to_end=False
    )
    
    # 对比系统
    comparison_df = eval_system.compare_systems()
    
    # 生成报告
    eval_system.generate_report()


if __name__ == "__main__":
    asyncio.run(test_rag_evaluation_system())
