# 智能文档问答助手

基于 LangGraph + FAISS 的多 Agent 文档问答系统，支持 PDF 上传、语义检索、会话记忆和历史会话恢复。

## 当前主线

当前推荐入口是 `main.py`。

- `main.py`：模块化主入口
- `agents/`：Research / Memory / General 三类 Agent
- `graph/`：Supervisor 与主图编排
- `memory/`：FAISS 向量存储、长期记忆、窗口压缩
- `document/`：PDF 解析、切分、入库、文档元数据注册
- `session/`：SQLite 会话持久化
- `ui/`：Gradio 界面

当前主线实现以 `main.py` 为入口，不再保留旧版单体/历史多 Agent 参考实现。

## 技术栈

| 模块 | 技术 |
|------|------|
| 工作流编排 | LangGraph |
| 向量数据库 | FAISS |
| LLM / Embedding | 智谱 GLM-5 / GLM-4-Flash / Embedding-3 |
| PDF 解析 | PyMuPDF4LLM |
| 前端 UI | Gradio |
| 会话持久化 | SQLite |

## 检索与记忆

- 文档知识库使用 FAISS 保存 `pdf_knowledge`
- 长期记忆使用 FAISS 保存 `user_semantic_memory`
- 文档切分采用父子块策略：
  - 父块：`chunk_size=1500`，`chunk_overlap=200`
  - 子块：`chunk_size=400`，`chunk_overlap=50`
- 检索时先召回子块，再通过 `parent_id + parent_text` 回溯父块内容
- 长期记忆按 `user_id + session_id` 做 metadata 过滤

## FAISS 配置

项目通过以下环境变量控制 FAISS：

- `FAISS_INDEX_ROOT`：索引和文档元数据保存目录，默认 `vectorstores`
- `FAISS_USE_GPU`：是否优先使用 GPU，`1` 表示开启
- `FAISS_GPU_DEVICE`：GPU 设备编号，默认 `0`

如果 GPU 初始化失败，代码会自动回退到 CPU 模式。

## 环境变量

`.env` / `.env.example` 中需要的主要配置：

```env
LLM_MODEL_ID=glm-5
LLM_API_KEY=your_key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

ZHIPU_API_KEY=your_key
ZHIPU_URL=https://open.bigmodel.cn/api/paas/v4/

FAISS_INDEX_ROOT=vectorstores
FAISS_USE_GPU=1
FAISS_GPU_DEVICE=0
```

## 启动

```bash
pip install -r requirements.txt
python main.py
```

默认访问地址：

`http://localhost:7861`

## 说明

- 会话历史由 `langgraph-checkpoint-sqlite` 持久化到本地 SQLite
- FAISS 索引和文档元数据默认持久化到 `vectorstores/`
- `requirements.txt` 已按平台区分：Windows 默认安装 `faiss-cpu`，非 Windows 默认安装 `faiss-gpu`
