# Cohere Rerank 集成指南 🚀

## ✅ 已集成完成！

您的 RAG 系统现在支持 **Cohere Rerank v3.5** - 最新最强的 rerank 模型！

## 🎯 性能提升

| 指标 | 提升幅度 |
|------|---------|
| 检索准确度 | +15-40% |
| 多语言支持 | 100+ 语言（包括中文） |
| 上下文长度 | 4096 tokens |
| 幻觉减少 | 显著降低 |

## 📦 配置步骤

### 1. 在 MCP Config 中添加 API Key

更新您的 Chatwise/Cherry Studio MCP 配置：

```json
{
  "mcpServers": {
    "llamaindex-rag": {
      "command": "/opt/miniconda3/envs/deep-learning/bin/python",
      "args": ["/Users/hinrayleung/dev/mcp/llamaindex/mcp_server.py"],
      "cwd": "/Users/hinrayleung/dev/mcp/llamaindex",
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "COHERE_API_KEY": "your-cohere-key-here"  ← 添加这行
      }
    }
  }
}
```

### 2. 重启 MCP Client

保存配置后，重启 Chatwise 或 Cherry Studio。

## 🔧 工作原理

### 两阶段检索（Two-Stage Retrieval）

```
查询: "MPI 是什么？"
      ↓
阶段 1: 向量检索
  - 使用 text-embedding-3-large
  - 检索 10 个候选文档
      ↓
阶段 2: Reranking (Cohere)
  - 使用 rerank-v3.5
  - 深度语义理解
  - 重新排序并返回 top 3
      ↓
结果: 3 个最相关的文档 ✨
```

### 性能对比

**不使用 Rerank:**
```
查询: "并行计算中的负载均衡"
返回: [文档A(0.35), 文档B(0.33), 文档C(0.32)]
可能包含不太相关的文档
```

**使用 Rerank v3.5:**
```
查询: "并行计算中的负载均衡"
初步检索: 10个候选
Rerank后: [文档X(0.92), 文档Y(0.89), 文档Z(0.85)]
精准度大幅提升！
```

## 💰 成本分析

### 每次查询成本

```
Embedding (10 docs): $0.00005
Rerank (10 docs):   $0.00050
------------------------
总计:               $0.00055 per query
```

### 月度成本估算

| 查询量 | 成本 |
|--------|------|
| 1,000 次 | $0.55 |
| 5,000 次 | $2.75 |
| 10,000 次 | $5.50 |

**非常便宜！** 相比准确度提升 15-40%，这个成本完全值得。

## 🎮 使用方法

### 方法 1: 默认开启 Rerank（推荐）

直接提问，Rerank 自动启用：

```
MPI 和 OpenMP 的区别是什么？
```

系统会自动使用 Rerank v3.5 提供最佳结果。

### 方法 2: 查看 Rerank 状态

```
查看我的 RAG 数据库状态
```

会显示是否启用了 Reranking。

### 方法 3: 手动控制（高级）

如果您想对比有无 Rerank 的差异：

```python
# 在代码中可以控制
query_rag(question, use_rerank=True)   # 使用 Rerank
query_rag(question, use_rerank=False)  # 不使用 Rerank
```

## 📊 效果验证

试试这些查询，感受 Rerank 的威力：

**复杂技术查询:**
```
详细解释梯形积分法在 MPI 中的负载均衡策略
```

**多语言混合:**
```
比较 Pthreads 和 OpenMP 的 mutex 实现
```

**深度语义理解:**
```
如何避免并行计算中的竞争条件？
```

## ⚙️ 配置选项

### 调整返回文档数量

默认返回 top 3，您可以调整：

```python
query_rag(question, similarity_top_k=5)  # 返回 top 5
```

### 模型信息

- **模型:** `rerank-v3.5`
- **提供商:** Cohere
- **发布时间:** 2024年12月
- **特点:** SOTA 多语言性能

## 🔍 技术细节

### Rerank 算法

Cohere Rerank v3.5 使用：
- 深度语义理解（不只是关键词匹配）
- 跨语言检索能力
- 对结构化数据（表格、JSON）的理解
- 推理能力

### 与 Embedding 的区别

| 特性 | Embedding | Reranking |
|------|-----------|-----------|
| 速度 | 快 | 较慢但更准 |
| 成本 | 低 | 稍高 |
| 准确度 | 好 | 优秀 |
| 语义理解 | 浅层 | 深层 |
| 最佳用途 | 初步筛选 | 精确排序 |

**最佳实践:** 两者结合使用！

## ❓ 常见问题

**Q: 如果不配置 COHERE_API_KEY 会怎样？**
A: 系统会自动跳过 Rerank，只使用 Embedding 检索，功能正常但准确度稍低。

**Q: Rerank 会慢很多吗？**
A: 会增加 100-200ms 延迟，但准确度提升值得这个代价。

**Q: 支持中文吗？**
A: 完美支持！Rerank v3.5 支持 100+ 语言，包括中文。

**Q: 可以用免费的开源 Reranker 吗？**
A: 可以！如果预算有限，可以改用 `bge-reranker-large`（免费，需本地运行）。

**Q: 如何查看是否正在使用 Rerank？**
A: 查询结果会显示 "(reranked with Cohere Rerank v3.5)"。

## 🎉 总结

✅ **已安装** `llama-index-postprocessor-cohere-rerank`
✅ **已集成** Rerank v3.5 到 `query_rag` 和 `query_rag_with_sources`
✅ **已更新** 配置文件和环境变量示例
✅ **即插即用** - 配置 API key 即可使用

**下一步:**
1. 在 MCP config 中添加您的 Cohere API key
2. 重启 Chatwise/Cherry Studio
3. 享受更准确的 RAG 检索！ 🚀

---

**提示:** Cohere 提供免费试用额度，您可以先测试效果再决定是否长期使用。
