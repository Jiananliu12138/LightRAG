# 🎯 RAG 组件级评估系统

## 概述

这是一个**完整的 RAG 评估系统**,不仅能评估端到端性能,还能深入分析每个组件(分块、嵌入、检索、生成)的性能。

### ✨ 核心特性

- ✅ **组件级评估**: 独立评估分块、嵌入、检索、生成质量
- ✅ **端到端评估**: 使用 RAGAS 框架评估整体性能
- ✅ **灵活对比**: 支持不同配置、不同系统的性能对比
- ✅ **可视化报告**: 生成详细的 CSV 和 HTML 报告
- ✅ **易于扩展**: 模块化设计,轻松添加新的评估指标

---

## 🏗️ 系统架构

```
RAG 评估系统
├── component_evaluators/           # 组件评估器
│   ├── __init__.py
│   ├── chunking_evaluator.py      # 分块质量评估
│   ├── embedding_evaluator.py     # 嵌入质量评估
│   └── retrieval_evaluator.py     # 检索性能评估
│
├── rag_evaluator_system.py        # 完整评估系统
├── evaluate_lightrag_complete.py  # LightRAG 集成脚本
├── quick_start_example.py         # 快速入门示例
├── eval_rag_quality.py            # 端到端评估 (RAGAS)
│
└── 文档
    ├── README_EVALUATION_SYSTEM.md   # 本文件
    └── RAG_EVALUATION_GUIDE.md       # 详细使用指南
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install numpy pandas scikit-learn matplotlib ragas langchain
```

### 2️⃣ 运行快速示例

```bash
cd /home/ik2200-2025-g2/WorkZone/LightRAG/lightrag/evaluation
python quick_start_example.py
```

**输出示例:**
```
🚀 RAG 评估系统 - 快速入门示例

📊 分块方法对比
固定大小分块: 48.31%
按段落分块:   100.00%
按句子分块:   98.55%

🏆 最佳方法: 按段落分块

📈 检索质量指标
  • P@5: 40.00%
  • R@5: 100.00%
  • MRR: 1.0000

🎯 总体评分: 84.11%
```

### 3️⃣ 评估 LightRAG 系统

```bash
# 确保 LightRAG 服务运行在 http://localhost:9621
# 确保 Ollama 服务运行在 http://localhost:11434

python evaluate_lightrag_complete.py
```

---

## 📊 评估指标详解

### 分块评估 (Chunking)

| 指标 | 说明 | 最佳值 |
|------|------|--------|
| 语义完整性 | chunk 是否保持语义完整 | 越高越好 |
| 边界质量 | 分块边界是否合理 | 越高越好 |
| 大小一致性 | chunk 大小是否均匀 | 越高越好 |
| 信息密度 | chunk 信息含量 | 越高越好 |
| 覆盖率 | 是否覆盖原文档 | 越高越好 |

### 嵌入评估 (Embedding)

| 指标 | 说明 | 最佳值 |
|------|------|--------|
| 语义相似度保持 | 嵌入是否保持文本相似度 | 越高越好 |
| 主题区分度 | 不同主题是否能区分 | 越高越好 |
| 检索准确率 | 基于嵌入的检索准确性 | 越高越好 |

### 检索评估 (Retrieval)

| 指标 | 说明 | 公式 |
|------|------|------|
| Precision@K | 前K个结果的准确率 | TP/(TP+FP) |
| Recall@K | 前K个结果的召回率 | TP/(TP+FN) |
| MRR | 平均倒数排名 | 1/rank |
| NDCG@K | 归一化折损累积增益 | DCG/IDCG |

### 端到端评估 (RAGAS)

| 指标 | 说明 |
|------|------|
| Faithfulness | 答案忠实度 |
| Answer Relevancy | 答案相关性 |
| Context Recall | 上下文召回率 |
| Context Precision | 上下文精确度 |

---

## 📖 使用场景

### 场景 1: 优化分块策略

```python
from component_evaluators import ChunkingEvaluator

# 测试不同的分块大小
chunk_sizes = [128, 256, 512, 1024]
best_score = 0
best_size = 0

for size in chunk_sizes:
    chunks = create_chunks(document, size=size, overlap=size//5)
    metrics = await evaluator.evaluate(document, chunks)
    
    if metrics._overall_score() > best_score:
        best_score = metrics._overall_score()
        best_size = size

print(f"最佳分块大小: {best_size} (得分: {best_score:.2%})")
```

### 场景 2: 对比不同嵌入模型

```python
from component_evaluators import EmbeddingEvaluator

models = ["nomic-embed-text", "bge-m3", "text-embedding-3-small"]
results = {}

for model in models:
    embedding_func = create_embedding_func(model)
    evaluator = EmbeddingEvaluator(embedding_func)
    metrics = await evaluator.evaluate(test_pairs=test_data)
    results[model] = metrics._overall_score()

best_model = max(results, key=results.get)
print(f"最佳嵌入模型: {best_model} ({results[best_model]:.2%})")
```

### 场景 3: 优化检索参数

```python
from component_evaluators import RetrievalEvaluator

# 测试不同的 top_k
for k in [3, 5, 10, 15, 20]:
    retrieval_func = create_retrieval_func(top_k=k)
    evaluator = RetrievalEvaluator(retrieval_func)
    metrics = await evaluator.evaluate(test_queries=queries)
    
    print(f"K={k}: P@K={metrics.precision_at_k[k]:.2%}, " 
          f"R@K={metrics.recall_at_k[k]:.2%}, "
          f"MRR={metrics.mrr:.4f}")
```

### 场景 4: 对比不同 RAG 系统

```python
from rag_evaluator_system import RAGEvaluationSystem, RAGSystemConfig

eval_system = RAGEvaluationSystem()

# 评估 LightRAG
config_lightrag = RAGSystemConfig(
    name="LightRAG",
    chunking_method="fixed_size",
    chunk_size=512,
    embedding_model="nomic-embed-text",
    # ... 其他配置
)
result_lightrag = await eval_system.evaluate_system(config_lightrag, ...)

# 评估 LlamaIndex
config_llamaindex = RAGSystemConfig(
    name="LlamaIndex",
    # ... 配置
)
result_llamaindex = await eval_system.evaluate_system(config_llamaindex, ...)

# 生成对比报告
eval_system.compare_systems([result_lightrag, result_llamaindex])
```

---

## 📁 输出文件

评估完成后,会在输出目录生成以下文件:

```
lightrag_evaluation_results/
├── LightRAG_Small_Chunks_20260121_143022.json  # 详细评估结果
├── comparison_20260121_143200.csv              # 对比表格
└── report_20260121_143200.html                 # HTML 报告
```

### JSON 结果示例

```json
{
  "config": {
    "name": "LightRAG_Small_Chunks_256",
    "chunking_method": "fixed_size",
    "chunk_size": 256,
    "embedding_model": "nomic-embed-text",
    "embedding_dim": 768,
    "retrieval_method": "hybrid",
    "top_k": 10,
    "llm_model": "qwen2.5:7b-instruct"
  },
  "chunking_metrics": {
    "semantic_completeness": 0.85,
    "boundary_quality": 0.92,
    "size_consistency": 0.78,
    "information_density": 0.88,
    "coverage": 0.95,
    "overall_score": 0.8760
  },
  "embedding_metrics": {
    "semantic_similarity_preservation": 0.82,
    "topic_separation": 0.78,
    "retrieval_accuracy": 0.90,
    "overall_score": 0.8350
  },
  "retrieval_metrics": {
    "precision@5": 0.70,
    "recall@5": 0.75,
    "mrr": 0.85,
    "ndcg@5": 0.72,
    "overall_score": 0.7375
  },
  "end_to_end_metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_recall": 0.92,
    "context_precision": 0.88,
    "ragas_score": 0.8675
  },
  "timestamp": "2026-01-21T14:30:22.123456"
}
```

---

## 🔧 高级用法

### 自定义评估指标

您可以扩展现有的评估器:

```python
from component_evaluators import ChunkingEvaluator

class MyCustomChunkingEvaluator(ChunkingEvaluator):
    def _evaluate_custom_metric(self, chunks):
        """您的自定义评估逻辑"""
        score = 0.0
        # ... 计算逻辑
        return score
    
    async def evaluate(self, original_document, chunks):
        # 调用父类方法
        metrics = await super().evaluate(original_document, chunks)
        
        # 添加自定义指标
        custom_score = self._evaluate_custom_metric(chunks)
        
        return metrics
```

### 集成到 CI/CD 流程

```bash
#!/bin/bash
# ci_evaluate.sh

# 运行评估
python evaluate_lightrag_complete.py

# 检查评估结果
SCORE=$(jq '.end_to_end_metrics.ragas_score' results/latest.json)

if (( $(echo "$SCORE < 0.8" | bc -l) )); then
    echo "❌ RAGAS 分数过低: $SCORE < 0.8"
    exit 1
fi

echo "✅ 评估通过: RAGAS 分数 = $SCORE"
```

---

## 💡 最佳实践

### 1. 建立基线

```python
# 第一次评估时建立基线
baseline_result = await eval_system.evaluate_system(current_config, ...)

# 保存基线
with open("baseline.json", "w") as f:
    json.dump(baseline_result.to_dict(), f)

# 后续评估时对比基线
current_result = await eval_system.evaluate_system(new_config, ...)
compare_with_baseline(baseline_result, current_result)
```

### 2. 逐组件优化

```
优化流程:
1. 运行完整评估,找出瓶颈
2. 优化单个组件(如分块)
3. 重新评估该组件
4. 评估对端到端性能的影响
5. 重复步骤 2-4 直到满意
```

### 3. A/B 测试

```python
# 配置 A
config_a = RAGSystemConfig(name="Config_A", chunk_size=256, ...)

# 配置 B
config_b = RAGSystemConfig(name="Config_B", chunk_size=512, ...)

# 评估两个配置
result_a = await eval_system.evaluate_system(config_a, ...)
result_b = await eval_system.evaluate_system(config_b, ...)

# 对比
eval_system.compare_systems([result_a, result_b])
```

### 4. 定期监控

```python
# 设置定期评估任务 (如每周)
import schedule

def weekly_evaluation():
    result = await eval_system.evaluate_system(...)
    save_to_monitoring_db(result)
    check_performance_degradation(result)

schedule.every().monday.at("02:00").do(weekly_evaluation)
```

---

## 🔍 故障排查

### 问题 1: 分块评分低

**可能原因:**
- 边界切断句子
- Chunk 大小不一致

**解决方案:**
```python
# 使用句子边界分块
chunks = chunk_by_sentence(doc, max_sentences=3)

# 或调整参数
chunks = chunk_fixed_size(doc, size=512, overlap=100)
```

### 问题 2: 检索评分低

**可能原因:**
- 嵌入模型不适合领域
- Top-K 不合适

**解决方案:**
```python
# 尝试不同嵌入模型
for model in ["nomic-embed-text", "bge-m3"]:
    test_embedding_model(model)

# 调整 top_k
for k in [3, 5, 10, 15]:
    test_retrieval_with_k(k)
```

### 问题 3: 端到端评分低但组件评分高

**可能原因:**
- LLM 生成能力不足
- Prompt 设计不佳

**解决方案:**
```python
# 升级 LLM 模型
config.llm_model = "qwen2.5:14b-instruct"  # 更大的模型

# 优化 Prompt
# 修改 LightRAG 的 prompt 模板
```

---

## 📚 相关资源

- [详细使用指南](RAG_EVALUATION_GUIDE.md)
- [RAGAS 官方文档](https://docs.ragas.io/)
- [LightRAG 项目](https://github.com/HKUDS/LightRAG)

---

## 🤝 贡献

欢迎贡献新的评估指标和功能！

### 添加新的评估指标

1. 在相应的 evaluator 中添加方法
2. 更新 Metrics 数据类
3. 添加测试用例
4. 更新文档

---

## 📄 许可

MIT License

---

## ✨ 总结

这个评估系统为您提供了:

✅ **全面的性能洞察**: 从组件到系统的完整评估  
✅ **数据驱动的优化**: 基于客观指标做决策  
✅ **灵活的对比分析**: 轻松对比不同配置和系统  
✅ **持续监控能力**: 追踪系统性能变化

立即开始评估您的 RAG 系统:

```bash
python quick_start_example.py
```

---

**作者**: LightRAG 评估团队  
**版本**: 1.0.0  
**更新时间**: 2026-01-21
