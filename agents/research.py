from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import tool

from agents.base import build_sub_agent
from document.registry import format_doc_list, get_all_documents


ROUTER_TOP_N = 5

SYSTEM_PROMPT = (
    "You are a research assistant for loaded PDF papers.\n"
    "Use list_documents when you need to inspect available papers.\n"
    "Use search_pdf for content retrieval. If the user names a paper, pass its filename as source.\n"
    "For cross-paper questions, compare evidence from the retrieved parent contexts."
)


def _doc_route_text(doc: dict) -> str:
    return "\n".join(
        part
        for part in [
            str(doc.get("title", "")).strip(),
            str(doc.get("filename", "")).strip(),
            str(doc.get("summary", "")).strip(),
        ]
        if part
    )


def make_research_agent(llm, fast_llm, pdf_store, loaded_docs: list):
    @tool
    def list_documents() -> str:
        """List the papers available to the current session."""
        result = format_doc_list(loaded_docs if loaded_docs else None)
        print(f"  [ResearchAgent][TOOL] list_documents: {result[:100]}...")
        return result

    def route_sources(query: str) -> list[str]:
        available_sources = set(pdf_store.list_sources())
        candidate_sources = set(loaded_docs) if loaded_docs else available_sources
        candidate_sources &= available_sources

        docs = [doc for doc in get_all_documents() if doc.get("filename") in candidate_sources]
        source_texts = {
            doc["filename"]: _doc_route_text(doc)
            for doc in docs
            if doc.get("filename")
        }
        routed = pdf_store.rank_sources_by_text(query, source_texts, top_n=ROUTER_TOP_N)
        print(
            "  [ResearchAgent][ROUTER] "
            f"candidate_count={len(candidate_sources)}, routed documents: {routed}"
        )
        return routed

    @tool
    def search_pdf(query: str, source: str = "") -> str:
        """Search PDF content. Use source to limit retrieval to one filename."""
        print(f"  [ResearchAgent][TOOL] search_pdf start, query: {query}, source: {source or 'auto'}")

        available_sources = set(pdf_store.list_sources())
        if not available_sources:
            return "No indexed PDF documents are available."

        if source:
            if source not in available_sources:
                return f"Document {source} is not indexed."
            target_sources = [source]
        else:
            target_sources = route_sources(query)
            if not target_sources:
                return "No relevant indexed PDF documents were found for this question."

        mqe_prompt = (
            "Generate 3 short alternative search queries for retrieving academic paper passages.\n"
            "Return one query per line, without numbering.\n\n"
            f"Original question: {query}"
        )
        try:
            extended = fast_llm.invoke(mqe_prompt).content.splitlines()
        except Exception:
            extended = []
        search_queries = [query] + [q.strip() for q in extended if q.strip()]
        print(f"  [ResearchAgent][TOOL] expanded queries: {search_queries}")

        def search_one(single_query: str):
            docs = []
            for selected_source in target_sources:
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
        print(f"  [ResearchAgent][TOOL] recalled child chunks: {len(all_child_docs)}")

        parent_map = {}
        for doc in all_child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in parent_map:
                source_name = doc.metadata.get("source", "")
                parent_text = doc.metadata.get("parent_text", doc.page_content)
                parent_map[parent_id] = f"[{source_name}]\n{parent_text}"

        unique_parents = list(parent_map.values())
        if not unique_parents:
            return f"No relevant passages were found in: {', '.join(target_sources)}."

        rerank_prompt = (
            "Answer the question using only the retrieved paper contexts below. "
            "Cite the source filename when comparing papers.\n\n"
            f"Question: {query}\n\n"
            f"Contexts:\n{unique_parents[:8]}"
        )
        result = fast_llm.invoke(rerank_prompt).content
        print(f"  [ResearchAgent][TOOL] result length: {len(result)}")
        return result

    return build_sub_agent(
        llm,
        [list_documents, search_pdf],
        SYSTEM_PROMPT,
        name="ResearchAgent",
        max_tool_calls=3,
    )
