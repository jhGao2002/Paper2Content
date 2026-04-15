# Eval 使用说明

这个目录集中存放项目里的评测脚本。

## 文件说明

- `generate_eval_dataset.py`
  - 读取 `eval/papers_example/` 下的 PDF 论文
  - 调用 LLM 自动生成 `question + ground_truth`
  - 输出到 `result/eval_dataset.json`

- `step1_generate.py`
  - 读取 `result/eval_dataset.json`
  - 调用当前系统生成回答与检索上下文
  - 输出缓存到 `result/generated_answers_cache_v2.json`

- `step2_evaluate.py`
  - 读取 `result/generated_answers_cache_v2.json`
  - 使用 Ragas 做评测
  - 输出结果到 `result/experiment_results_log_v2.csv`

- `calculate_f1.py`
  - 读取 `result/experiment_results_log_v2.csv`
  - 计算 `retrieval_f1`
  - 输出到 `result/experiment_results_with_f1.csv`

- `eval_baseline.py`
  - 直接跑一套基线评测流程
  - 输出到 `result/experiment_results_log.csv`

## 推荐使用流程

在仓库根目录执行：

```powershell
python eval/step1_generate.py
python eval/step2_evaluate.py
python eval/calculate_f1.py
```

如果你要跑基线版本：

```powershell
python eval/eval_baseline.py
```

如果你要先自动构造评测集：

```powershell
python eval/generate_eval_dataset.py
```

常用参数示例：

```powershell
python eval/generate_eval_dataset.py --questions-per-paper 10 --max-chars 15000
```

## 前置条件

- 根目录存在 `result/eval_dataset.json`
- `.env` 中已经配置评测所需的模型相关环境变量
- 项目依赖已经安装完成

## 结果文件

- `result/generated_answers_cache_v2.json`
- `result/experiment_results_log_v2.csv`
- `result/experiment_results_with_f1.csv`
- `result/experiment_results_log.csv`

## 说明

- 这些脚本现在会自动按自身位置定位仓库根目录，所以推荐始终从仓库根目录执行。
- 评测产物统一放在根目录下的 `result/`，方便集中查看和后续清理。
- `eval/papers_example/` 用来临时存放待生成评测集的论文，目录中的实际论文文件不会被 Git 跟踪。

## `eval_dataset.json` 构造方案

当前项目不会自动生成 `result/eval_dataset.json`，需要手工整理评测问答集。现有脚本只要求每条样本至少包含下面两个字段：

- `question`：评测时实际提给系统的问题
- `ground_truth`：用于评测的标准答案

最小可用格式如下：

```json
[
  {
    "question": "LoRA 的核心思想是什么？",
    "ground_truth": "LoRA 通过冻结原始模型参数，仅训练低秩矩阵来近似权重更新，从而降低微调成本。"
  },
  {
    "question": "AdapterFusion 主要解决什么问题？",
    "ground_truth": "AdapterFusion 旨在融合多个任务适配器中的知识，在避免灾难性遗忘的同时提升迁移能力。"
  }
]
```

推荐构造流程：

1. 先确定评测范围，只选当前知识库里已经入库的论文或资料。
2. 每篇文档整理 5 到 10 个问题，覆盖定义类、方法类、对比类、实验结论类问题。
3. `ground_truth` 尽量写成简洁但完整的标准答案，不要只写关键词。
4. 避免把问题写得过于模糊，尽量让答案能够在单篇文档或少量上下文中定位。
5. 整理完成后保存到根目录 `result/eval_dataset.json`，再运行 `step1_generate.py` 和 `step2_evaluate.py`。

编写样本时的建议：

- 尽量避免“这篇文章好吗”这类开放问题，优先使用可核对事实的问题。
- 如果问题涉及多个方法对比，`ground_truth` 里要明确比较维度。
- 每条样本只评一个核心点，避免一个问题里同时追问多个子问题。
- 如果后续要分析召回效果，可以额外给每条样本补充 `source`、`keywords` 等字段，但当前脚本不会读取它们。
