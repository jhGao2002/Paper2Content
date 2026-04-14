from langchain_core.documents import Document


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """A lightweight local splitter to avoid heavyweight optional NLP deps."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    docs: list[Document] = []
    start = 0
    step = chunk_size - chunk_overlap
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            docs.append(Document(page_content=chunk))
        if end >= text_length:
            break
        start += step

    return docs
