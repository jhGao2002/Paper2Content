# Eval 使用说明

`eval/` 目录用于统一管理项目评测。现在的主入口是：

```powershell
python eval/eval.py
```

在放好测试论文后，只需要运行这一条命令，就会自动完成：

1. 检查或生成 `result/eval_dataset.json`
2. 运行 `Baseline_Naive_RAG`
3. 运行 `Current_Project_Full`
4. 使用 Ragas 计算评测指标
5. 输出中间文件到 `result/`
6. 将本次对比结果追加写入本文件末尾

## 目录说明

- `eval.py`
  - 统一评测入口
  - 自动串起数据集生成、baseline 评测、当前系统评测、Ragas 打分、结果汇总

- `generate_eval_dataset.py`
  - 从 `eval/papers_example/` 中的论文自动生成 `result/eval_dataset.json`
  - 也可以单独运行，用于只生成评测集

- `papers_example/`
  - 临时放评测论文的目录
  - 目录中的实际论文文件不会被 Git 跟踪

## 使用流程

1. 把待评测论文 PDF 放进 `eval/papers_example/`
2. 运行：

```powershell
python eval/eval.py
```

如果希望强制重新生成评测集：

```powershell
python eval/eval.py --regenerate-dataset
```

如果想调节自动生成评测集的规模：

```powershell
python eval/eval.py --questions-per-paper 10 --max-chars 15000
```

## 评测输入

默认输入包括两部分：

- 论文目录：`eval/papers_example/`
- 评测集：`result/eval_dataset.json`

如果 `result/eval_dataset.json` 不存在，`eval.py` 会先调用 LLM 自动生成它。  
自动生成只是冷启动方案，建议后续人工抽查一部分 `question + ground_truth`。

## Baseline 与当前系统的对比定义

### 1. `Baseline_Naive_RAG`

这是一个真正的朴素 RAG baseline，用于和当前项目做清晰对比。

参数与行为：

- 单层切块：`chunk_size=800`
- 重叠：`chunk_overlap=150`
- 单路查询：不做 query expansion
- 检索：直接向量召回 `top_k=4`
- 不做 rerank
- 不做 parent-child chunking
- 不做 Supervisor 路由
- 不做多 Agent 协同
- 不使用会话记忆
- 基于检索到的 chunk 直接让 LLM 生成答案

### 2. `Current_Project_Full`

这是当前项目的完整版本，也就是你现在主系统里的增强 RAG 流程。

参数与行为：

- 父块切分：`chunk_size=1500`
- 父块重叠：`chunk_overlap=200`
- 子块切分：`chunk_size=400`
- 子块重叠：`chunk_overlap=50`
- Query Expansion：原问题 + 3 个扩展查询
- 每个查询召回：`top_k=3`
- 先召回子块，再回溯父块上下文
- 使用 LLM 对候选父块做 rerank / 片段筛选
- 通过 `Supervisor` 做路由
- 支持多 Agent 协作
- 支持复杂问题的 plan-and-act
- 支持长期记忆能力

## 当前系统相对 Baseline 的主要优化点

- 从单层 chunk 升级为 parent-child chunking，尽量兼顾召回精度和回答上下文完整性
- 从单次查询升级为 query expansion，降低用户表述和论文措辞不一致导致的漏召回
- 增加 rerank，减少把相似但不关键的片段直接塞给生成模型
- 增加 Supervisor 与多 Agent 路由，允许复杂问题拆步处理
- 增加长期记忆能力，服务真实多轮对话场景

## 中间产物

评测运行后，`result/` 下会输出这些文件：

- `eval_dataset.json`
- `generated_answers_cache_baseline.json`
- `generated_answers_cache_v2.json`
- `experiment_results_log_baseline.csv`
- `experiment_results_with_f1_baseline.csv`
- `experiment_results_log_v2.csv`
- `experiment_results_with_f1.csv`
- `eval_comparison_summary.csv`
- `eval_comparison_summary.json`
- `eval_runtime/`

其中：

- `generated_answers_cache_baseline.json` 是 baseline 的回答缓存
- `generated_answers_cache_v2.json` 是当前系统的回答缓存
- `experiment_results_log_*.csv` 是 Ragas 原始分项结果
- `experiment_results_with_f1*.csv` 额外包含 `retrieval_f1`
- `eval_comparison_summary.*` 是两套方案的均值对比摘要
- `eval_runtime/` 是隔离运行时目录，避免污染主知识库和主会话数据

## 前置条件

- `.env` 中已配置模型相关环境变量
- 项目依赖已安装完成
- `eval/papers_example/` 下已有待评测论文，或 `result/eval_dataset.json` 已准备好

## 说明

- 评测脚本会使用隔离的运行时目录，不会污染你主流程使用的 `vectorstores/`、`documents.json`、`sessions.db`
- 生成的评测结果会额外追加写入本文件，便于直接查看历史记录

## 评测结果记录
