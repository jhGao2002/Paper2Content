# Eval 使用说明

`eval/` 目录用于统一管理 RAG 评测。当前主入口是：

```powershell
python eval/eval.py
```

放好测试论文之后，只需要运行这一条命令，就会自动完成：

1. 检查或生成评测集
2. 运行 4 组 RAG 变体
3. 使用 Ragas 打分
4. 统计生成阶段和评测阶段的 token 消耗
5. 输出清晰命名的中间产物
6. 把本次结果追加写入本文件末尾

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

如果想调节自动生成评测集的规模：

```powershell
python eval/eval.py --questions-per-paper 10 --max-chars 15000
```

## 当前评测设计

这次评测故意简化成两个变量：

- 是否使用父子块
- 是否使用 `Embedding-3`

因此总共有 4 组数据：

1. `01_baseline_flat_lexical`
   基线版本：单层切块 + 非 `Embedding-3`（词法哈希 embedding）
2. `02_flat_embedding3`
   只替换 embedding：单层切块 + `Embedding-3`
3. `03_parent_child_lexical`
   只替换分块：父子块 + 非 `Embedding-3`
4. `04_parent_child_embedding3`
   当前最强的这组检索配置：父子块 + `Embedding-3`

这里不再引入 query expansion、rerank、多 Agent 路由等额外变量，目的是把对比聚焦在：

- 分块策略本身是否有效
- embedding 方案本身是否有效

## 四组方案说明

### `01_baseline_flat_lexical`

- 单层切块：`chunk_size=800`
- overlap：`150`
- embedding：词法哈希 embedding
- 检索：FAISS top-k
- 不使用父子块

### `02_flat_embedding3`

- 单层切块：`chunk_size=800`
- overlap：`150`
- embedding：`Embedding-3`
- 检索：FAISS top-k
- 不使用父子块

### `03_parent_child_lexical`

- 父块：`chunk_size=1500`
- 父块 overlap：`200`
- 子块：`chunk_size=400`
- 子块 overlap：`50`
- embedding：词法哈希 embedding
- 检索时先搜子块，再回溯父块

### `04_parent_child_embedding3`

- 父块：`chunk_size=1500`
- 父块 overlap：`200`
- 子块：`chunk_size=400`
- 子块 overlap：`50`
- embedding：`Embedding-3`
- 检索时先搜子块，再回溯父块

## 评测指标

当前会统一输出这些指标：

- `context_precision`
- `context_recall`
- `retrieval_f1`
- `answer_correctness`
- `answer_relevancy`
- `faithfulness`

其中：

- `retrieval_f1` 是由 `context_precision` 和 `context_recall` 计算得到
- `answer_relevancy` 用来看回答是否真正切题
- `faithfulness` 用来看回答是否忠于检索上下文

## Token 统计

当前结果里会记录两类 token：

- `generation_*_tokens`
  指每个变体在“检索后生成答案”阶段的 token 用量
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
result/eval_runs/run_20260415_183000/
```

同时还会同步一份最新结果到：

```text
result/eval_runs/latest/
```

## 中间产物命名

每次运行的目录结构如下：

```text
dataset/
  eval_dataset.json

answers/
  01_baseline_flat_lexical.json
  02_flat_embedding3.json
  03_parent_child_lexical.json
  04_parent_child_embedding3.json

metrics/
  01_baseline_flat_lexical.csv
  02_flat_embedding3.csv
  03_parent_child_lexical.csv
  04_parent_child_embedding3.csv

summaries/
  variant_metrics_summary.csv
  variant_comparison.csv
  variant_token_summary.csv
  run_manifest.json

runtime/
  ...
```

各文件含义：

- `dataset/eval_dataset.json`
  本次评测实际使用的数据集副本
- `answers/*.json`
  每个变体生成的回答和检索上下文
- `metrics/*.csv`
  每个变体逐样本的 Ragas 结果
- `summaries/variant_metrics_summary.csv`
  4 组方案的均值指标汇总
- `summaries/variant_comparison.csv`
  便于横向比较的总表
- `summaries/variant_token_summary.csv`
  4 组方案的 token 汇总
- `summaries/run_manifest.json`
  本次运行的完整元信息

## 前置条件

- `.env` 中已配置模型相关环境变量
- 项目依赖已安装完成
- `eval/papers_example/` 下已有待评测论文，或 `result/eval_dataset.json` 已准备好

## 说明

- `eval/papers_example/` 里的实际论文不会被 Git 跟踪
- 每次评测结束后，本文件末尾会追加一段结果记录

## 评测结果记录
