# Eval 使用说明

这个目录集中存放项目里的评测脚本。

## 文件说明

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
