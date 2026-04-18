import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(default) if os.getenv(name) is None else os.getenv(name, str(default))
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip()


def _parse_csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_json_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} 必须是合法 JSON 对象") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{name} 必须是 JSON 对象")
    return {str(key): value for key, value in data.items()}


def _configure_langsmith_env() -> None:
    tracing_enabled = _env_flag("LANGSMITH_TRACING", False)
    os.environ["LANGSMITH_TRACING"] = "true" if tracing_enabled else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing_enabled else "false"

    for name in ("LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT", "LANGSMITH_WORKSPACE_ID"):
        value = _env_text(name)
        if value:
            os.environ[name] = value

    project = _env_text("LANGSMITH_PROJECT")
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project


def is_langsmith_enabled() -> bool:
    return _env_flag("LANGSMITH_TRACING", False)


def get_langsmith_project() -> str:
    return _env_text("LANGSMITH_PROJECT", "paper2content")


def get_langsmith_tags(*extra_tags: str) -> list[str]:
    tags = list(_parse_csv_env("LANGSMITH_TAGS"))
    tags.extend(tag.strip() for tag in extra_tags if isinstance(tag, str) and tag.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


def get_langsmith_metadata(extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {
        "app": _env_text("LANGSMITH_APP_NAME", "paper_assistant"),
        "project": get_langsmith_project(),
        "tracing_enabled": is_langsmith_enabled(),
    }
    metadata.update(_parse_json_env("LANGSMITH_METADATA"))
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def build_langsmith_runnable_config(
    *,
    run_name: str | None = None,
    extra_tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    configurable: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if run_name:
        config["run_name"] = run_name

    tags = get_langsmith_tags(*(extra_tags or []))
    if tags:
        config["tags"] = tags

    metadata = get_langsmith_metadata(extra_metadata)
    if metadata:
        config["metadata"] = metadata

    if configurable:
        config["configurable"] = configurable

    return config


def with_langsmith_config(
    runnable: Any,
    *,
    run_name: str | None = None,
    extra_tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
):
    config = build_langsmith_runnable_config(
        run_name=run_name,
        extra_tags=extra_tags,
        extra_metadata=extra_metadata,
    )
    if not config:
        return runnable
    return runnable.with_config(config)


_configure_langsmith_env()


def get_llm():
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    return with_langsmith_config(
        llm,
        run_name="primary_llm",
        extra_tags=["llm", "primary"],
        extra_metadata={
            "component": "llm",
            "model_role": "primary",
            "model_name": os.getenv("LLM_MODEL_ID"),
        },
    )


def get_fast_llm():
    llm = ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        temperature=1.0,
        max_tokens=65536,
    )
    return with_langsmith_config(
        llm,
        run_name="fast_llm",
        extra_tags=["llm", "fast"],
        extra_metadata={
            "component": "llm",
            "model_role": "fast",
            "model_name": "glm-4-flash",
        },
    )


def get_embeddings():
    return OpenAIEmbeddings(
        model="Embedding-3",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_URL"),
        check_embedding_ctx_length=False,
    )
