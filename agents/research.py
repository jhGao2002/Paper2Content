from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import tool

from agents.base import build_sub_agent
from document.registry import format_doc_list


SYSTEM_PROMPT = (
    "你是文档检索专家。\n"
    "工作要求：\n"
    "1. 先判断是否需要查看当前已加载文档，可使用 list_documents。\n"
    "2. 需要检索内容时使用 search_pdf，可通过 source 参数限定某篇文档。\n"
    "3. 只依据检索到的内容回答，不要编造论文细节。\n"
    "如果没有足够证据，请明确说明。"
)


def make_research_agent(llm, fast_llm, pdf_store, loaded_docs: list):
    @tool
    def list_documents() -> str:
        """列出当前可检索的文档。"""
        result = format_doc_list(loaded_docs)
        if "暂无文档" in result:
            try:
                sources = pdf_store.list_sources()
                if sources:
                    doc_list = "\n".join(f"- {name}" for name in sources)
                    result = (
                        f"当前共检测到 {len(sources)} 篇已入库文档：\n"
                        f"{doc_list}\n\n"
                        "你可以继续指定文档名做定向检索。"
                    )
            except Exception as exc:
                print(f"  [ResearchAgent][TOOL] FAISS 文档列表读取失败: {exc}")

        print(f"  [ResearchAgent][TOOL] list_documents: {result[:100]}...")
        return result

    @tool
    def search_pdf(query: str, source: str = "") -> str:
        """在 PDF 知识库中检索相关内容，可选按 source 限定文档。"""
        print(f"  [ResearchAgent][TOOL] search_pdf 开始，query: {query}, source: {source or '全部'}")
        if not loaded_docs:
            return "当前会话未选择任何文档，请先在会话面板勾选要加载的文档。"
        if source and source not in loaded_docs:
            return f"文档 {source} 未加入当前会话，请先在会话面板勾选该文档。"
        mqe_prompt = (
            "请围绕下面的问题补充 3 个不同表达方式的检索子查询，"
            "每行一个，避免重复。\n"
            f"问题：{query}"
        )
        extended = fast_llm.invoke(mqe_prompt).content.split("\n")
        search_queries = [query] + [q.strip() for q in extended if q.strip()]
        print(f"  [ResearchAgent][TOOL] 扩展查询: {search_queries}")

        def search_one(single_query: str):
            if source:
                return pdf_store.similarity_search(
                    single_query,
                    k=3,
                    filter={"must": [{"key": "source", "match": {"value": source}}]},
                )
            docs = []
            for selected_source in loaded_docs:
                docs.extend(
                    pdf_store.similarity_search(
                        single_query,
                        k=3,
                        filter={"must": [{"key": "source", "match": {"value": selected_source}}]},
                    )
                )
            return docs

        with ThreadPoolExecutor() as executor:
            batches = list(executor.map(search_one, search_queries))
        all_child_docs = [doc for batch in batches for doc in batch]
        print(f"  [ResearchAgent][TOOL] 子块召回数: {len(all_child_docs)}")

        parent_map = {}
        for doc in all_child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_map:
                parent_map[parent_id] = doc.metadata.get("parent_text", doc.page_content)

        unique_parents = list(parent_map.values())
        if not unique_parents:
            if source:
                return f"没有在文档 {source} 中检索到相关内容。"
            return "没有检索到相关文档内容。"

        rerank_prompt = (
            f"请从下面的候选片段中挑选与问题最相关的内容，并直接返回可用于回答的片段。\n\n"
            f"问题：{query}\n\n"
            f"候选片段：\n{unique_parents[:8]}"
        )
        result = fast_llm.invoke(rerank_prompt).content
        print(f"  [ResearchAgent][TOOL] 检索结果长度: {len(result)}")
        return result

    return build_sub_agent(
        llm,
        [list_documents, search_pdf],
        SYSTEM_PROMPT,
        name="ResearchAgent",
        max_tool_calls=3,
    )
