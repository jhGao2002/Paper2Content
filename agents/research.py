from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import tool

from agents.base import build_sub_agent
from document.registry import format_doc_list

SYSTEM_PROMPT = (
    "你是专业的学术文档检索专家。\n"
    "【工作流程】：\n"
    "1. 如果不确定知识库中有哪些文档，先调用 list_documents 查看。\n"
    "2. 使用 search_pdf 检索相关内容，可通过 source 参数限定在某篇文档中搜索。\n"
    "3. 基于检索结果忠实回答，不知道就明确说不知道。\n"
    "【核心纪律】：只回答用户最新的问题，不重复回答历史问题。"
)


def make_research_agent(llm, fast_llm, pdf_store, loaded_docs: list):
    @tool
    def list_documents() -> str:
        """列出知识库中已入库的文档。"""
        result = format_doc_list()
        if "暂无文档" in result:
            try:
                sources = pdf_store.list_sources()
                if sources:
                    doc_list = "\n".join(f"- {name}" for name in sources)
                    result = (
                        f"知识库中检测到 {len(sources)} 个文档（元数据未完整注册，仅显示文件名）：\n"
                        f"{doc_list}\n\n"
                        "提示：重新上传这些文档可生成标题和摘要。"
                    )
            except Exception as exc:
                print(f"  [ResearchAgent][TOOL] FAISS 文档扫描失败: {exc}")

        print(f"  [ResearchAgent][TOOL] list_documents: {result[:100]}...")
        return result

    @tool
    def search_pdf(query: str, source: str = "") -> str:
        """在 PDF 知识库中检索相关内容，可按 source 过滤指定文档。"""
        print(f"  [ResearchAgent][TOOL] search_pdf 开始，query: {query}, source: {source or '全部'}")
        mqe_prompt = (
            "为了在学术文档中更全面地检索下面的问题，请生成 3 个不同表达或侧重点的相似搜索词。"
            "不要加序号，每行一个。\n"
            f"原始问题：{query}"
        )
        extended = fast_llm.invoke(mqe_prompt).content.split("\n")
        search_queries = [query] + [q.strip() for q in extended if q.strip()]
        print(f"  [ResearchAgent][TOOL] 扩展查询词: {search_queries}")

        def search_one(single_query: str):
            if source:
                return pdf_store.similarity_search(
                    single_query,
                    k=3,
                    filter={"must": [{"key": "source", "match": {"value": source}}]},
                )
            return pdf_store.similarity_search(single_query, k=3)

        with ThreadPoolExecutor() as executor:
            batches = list(executor.map(search_one, search_queries))
        all_child_docs = [doc for batch in batches for doc in batch]
        print(f"  [ResearchAgent][TOOL] 召回子块数: {len(all_child_docs)}")

        parent_map = {}
        for doc in all_child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_map:
                parent_map[parent_id] = doc.metadata.get("parent_text", doc.page_content)

        unique_parents = list(parent_map.values())
        if not unique_parents:
            if source:
                return f"未能在《{source}》中检索到相关内容。"
            return "未能在知识库中检索到相关内容。"

        rerank_prompt = (
            f"以下是从文档中粗筛出的若干片段。请挑选与问题【{query}】最相关的片段并拼接，删除无关内容。\n\n"
            f"片段：\n{unique_parents[:8]}"
        )
        result = fast_llm.invoke(rerank_prompt).content
        print(f"  [ResearchAgent][TOOL] 检索完成，结果长度: {len(result)}")
        return result

    return build_sub_agent(
        llm,
        [list_documents, search_pdf],
        SYSTEM_PROMPT,
        name="ResearchAgent",
        max_tool_calls=3,
    )
