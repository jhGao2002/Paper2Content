# Eval 使用说明

`eval/` 目录用于统一管理当前项目的 RAG 评测。主入口是：

```powershell
python eval/eval.py
```

运行后会自动完成：

1. 检查或生成评测集
2. 运行 3 组检索变体
3. 使用 Ragas 打分
4. 统计生成阶段和评测阶段的 token 消耗
5. 输出清晰命名的中间产物
6. 将本次结果追加写入本文件末尾

## 使用流程

1. 把待评测论文 PDF 放进 `eval/papers_example/`
2. 运行：

```powershell
python eval/eval.py
```

如果想强制重新生成评测集：

```powershell
python eval/eval.py --regenerate-dataset
```

## 默认参数

正式测试默认参数：

- 每篇论文生成 `5` 个问答
- 使用 `eval/papers_example/` 下全部论文

beta 测试默认参数：

- 通过 `--beta` 开启
- 只读取 `eval/papers_example/` 中按文件名排序后的第一篇论文
- 默认最多保留 `2` 条问答样本

示例：

```powershell
python eval/eval.py --beta
```

## 当前评测设计

当前只对比 3 组变体：

1. `01_fixed_chunk_embedding3`
   固定大小分块 + `Embedding-3`
2. `02_parent_child_lexical`
   当前项目的父子块分块 + 词法哈希 embedding
3. `03_parent_child_embedding3`
   当前项目的父子块分块 + `Embedding-3`

这样对比的目的很明确：

- 先看“固定分块”到“父子块分块”是否带来收益
- 再看在当前项目分块策略下，词法哈希 embedding 升级到 `Embedding-3` 是否带来收益

## 三组方案说明

### `01_fixed_chunk_embedding3`

- 单层切块：`chunk_size=800`
- overlap：`150`
- embedding：`Embedding-3`
- 检索：FAISS top-k

### `02_parent_child_lexical`

- 父块：`chunk_size=1500`
- 父块 overlap：`200`
- 子块：`chunk_size=400`
- 子块 overlap：`50`
- embedding：词法哈希 embedding
- 检索时先搜子块，再回溯父块

### `03_parent_child_embedding3`

- 父块：`chunk_size=1500`
- 父块 overlap：`200`
- 子块：`chunk_size=400`
- 子块 overlap：`50`
- embedding：`Embedding-3`
- 检索时先搜子块，再回溯父块

## 评测指标

当前统一输出这些指标：

- `context_precision`
- `context_recall`
- `retrieval_f1`
- `answer_correctness`
- `faithfulness`

### `context_precision`

衡量“检索出来的上下文里，有多少是真正有用的”。

- 分数越高，说明召回噪声越少

### `context_recall`

衡量“回答所需的关键信息，是否已经被检索出来”。

- 分数越高，说明证据覆盖更充分

### `retrieval_f1`

这是一定会计算的核心指标，用来综合看召回质量。

计算公式：

```text
retrieval_f1 = 2 * context_precision * context_recall / (context_precision + context_recall)
```

特殊情况：

- 如果 `context_precision + context_recall = 0`
- 或某一项缺失

则按 `0` 处理。

### `answer_correctness`

衡量最终回答与标准答案 `ground_truth` 的一致程度。

- 分数越高，说明回答越接近标准答案

### `faithfulness`

衡量回答是否忠于检索上下文。

- 分数越高，说明回答更受检索证据支持

## Token 统计

当前结果里会记录两类 token：

- `generation_*_tokens`
  指某个变体在“检索后生成答案”阶段的 token 用量
- `judge_*_tokens`
  指 Ragas 评测阶段的 token 用量

注意：

- 当前统计的是 LLM token
- embedding 阶段的 token 或计费不在这份统计里

## 输出目录

评测结果会输出到：

```text
result/eval_runs/
```

每次运行都会生成一个新目录，例如：

```text
result/eval_runs/full_20260415_210000/
```

或 beta 模式下：

```text
result/eval_runs/beta_20260415_210000/
```

同时还会同步一份最新结果到：

```text
result/eval_runs/latest/
```

## 中间产物命名

每个变体会单独占一个文件夹，命名格式是：

```text
变体名_时间戳
```

例如：

```text
01_fixed_chunk_embedding3_20260415_210000/
```

运行目录结构示例如下：

```text
full_20260415_210000/
  dataset/
    eval_dataset.json
  01_fixed_chunk_embedding3_20260415_210000/
    answers/
      answers.json
    metrics/
      metrics.csv
    runtime/
      ...
  02_parent_child_lexical_20260415_210000/
    ...
  03_parent_child_embedding3_20260415_210000/
    ...
  summaries/
    variant_metrics_summary.csv
    variant_token_summary.csv
    run_manifest.json
```

## 断点续跑

当前支持断点续跑：

- 如果某个变体的 `answers/answers.json` 已存在，就复用回答缓存
- 如果某个变体的 `metrics/metrics.csv` 已存在，就复用评测结果

## 前置条件

- `.env` 中已配置模型相关环境变量
- 项目依赖已安装完成
- `eval/papers_example/` 下已有待评测论文，或正式模式所需的 `result/eval_dataset.json` 已准备好

## 说明

- `eval/papers_example/` 里的实际论文不会被 Git 跟踪
- 每次评测结束后，本文件末尾会追加一段结果记录
- 结果记录使用 Markdown 表格

## 评测结果记录
