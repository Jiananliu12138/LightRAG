import asyncio
import json
import sys
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056789/Rag_Chunking/tiktoken_cache"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class OpenAICompatibleLLMConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    timeout: int = 120
    temperature: float | None = 0.2
    max_tokens: int | None = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    embedding_dim: int = 1024
    max_token_size: int = 8192
    timeout: int = 120


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    binding: str = "cohere"
    model: str = "BAAI/bge-reranker-v2-m3"
    base_url: str = "http://127.0.0.1:8002/rerank"
    api_key: str = "EMPTY"
    min_score: float = 0.0
    enable_chunking: bool = False
    max_tokens_per_doc: int = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    input_path: str | None = None
    question: str = "What is LightRAG?"
    mode: str = "hybrid"
    working_dir: str = str(Path(__file__).resolve().parent / "rag_storage"/"2wikimqa")
    output_path: str = str(Path(__file__).resolve().parent / "query_result.json")
    llm: OpenAICompatibleLLMConfig = field(
        default_factory=lambda: OpenAICompatibleLLMConfig(
            model="Qwen2.5-7B-Instruct",
            base_url="http://127.0.0.1:8005/v1",
            api_key="EMPTY",
            temperature=0.2,
            max_tokens=4096,
            extra_body={},
        )
    )
    embedding: OpenAICompatibleEmbeddingConfig = field(
        default_factory=lambda: OpenAICompatibleEmbeddingConfig(
            model="BAAI/bge-large-en-v1.5",
            base_url="http://127.0.0.1:8003/v1",
            api_key="EMPTY",
            embedding_dim=1024,
            max_token_size=8192,
        )
    )
    rerank: RerankConfig = field(default_factory=RerankConfig)


CONFIG = RunConfig()


def is_custom_chunk_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and "splits" in payload


def load_custom_chunk_payloads(config: RunConfig) -> list[dict[str, Any]]:
    if not config.input_path:
        return []

    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        payloads: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not is_custom_chunk_payload(payload):
                    raise ValueError(
                        f"JSONL line {line_number} is not a custom chunk payload with 'splits'"
                    )
                payloads.append(payload)
        return payloads

    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if is_custom_chunk_payload(payload):
            return [payload]
        if isinstance(payload, list) and all(is_custom_chunk_payload(item) for item in payload):
            return payload
        return []

    return []


def extract_chunk_doc_id(raw_chunk: Any, default_doc_id: str | None = None) -> str | None:
    if isinstance(raw_chunk, dict):
        return raw_chunk.get("doc_id") or raw_chunk.get("full_doc_id") or default_doc_id
    if isinstance(raw_chunk, (list, tuple)):
        if len(raw_chunk) > 1 and raw_chunk[1] not in (None, ""):
            return str(raw_chunk[1])
        return default_doc_id
    return default_doc_id


def summarize_custom_chunk_payloads(payloads: list[dict[str, Any]]) -> tuple[int, int, int]:
    payload_count = len(payloads)
    chunk_count = 0
    doc_ids: set[str] = set()

    for payload in payloads:
        payload_doc_id = payload.get("doc_id") or payload.get("full_doc_id")
        splits = payload.get("splits", [])
        chunk_count += len(splits)

        if payload_doc_id not in (None, ""):
            doc_ids.add(str(payload_doc_id))

        for raw_chunk in splits:
            chunk_doc_id = extract_chunk_doc_id(raw_chunk, payload_doc_id)
            if chunk_doc_id not in (None, ""):
                doc_ids.add(str(chunk_doc_id))

    return payload_count, len(doc_ids), chunk_count

async def local_llm_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    from lightrag.llm.openai import openai_complete_if_cache

    llm_config = CONFIG.llm
    request_kwargs: dict[str, Any] = {}
    if llm_config.temperature is not None:
        request_kwargs["temperature"] = llm_config.temperature
    if llm_config.max_tokens is not None:
        request_kwargs["max_tokens"] = llm_config.max_tokens
    if llm_config.extra_body:
        request_kwargs["extra_body"] = llm_config.extra_body
    request_kwargs.update(kwargs)

    return await openai_complete_if_cache(
        model=llm_config.model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        timeout=llm_config.timeout,
        **request_kwargs,
    )


def build_rerank_model_func(config: RunConfig):
    if not config.rerank.enabled:
        return None

    from lightrag.rerank import ali_rerank, cohere_rerank, jina_rerank

    rerank_functions = {
        "cohere": cohere_rerank,
        "jina": jina_rerank,
        "aliyun": ali_rerank,
    }
    selected_rerank_func = rerank_functions.get(config.rerank.binding)
    if selected_rerank_func is None:
        raise ValueError(
            f"Unsupported rerank binding: {config.rerank.binding}. "
            "Use one of: cohere, jina, aliyun."
        )

    async def local_rerank(
        query: str,
        documents: list[str],
        top_n: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        merged_extra_body = dict(config.rerank.extra_body)
        if extra_body:
            merged_extra_body.update(extra_body)

        request_kwargs: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "api_key": config.rerank.api_key,
            "model": config.rerank.model,
            "base_url": config.rerank.base_url,
            "extra_body": merged_extra_body or None,
        }

        if config.rerank.binding == "cohere":
            request_kwargs["enable_chunking"] = config.rerank.enable_chunking
            request_kwargs["max_tokens_per_doc"] = config.rerank.max_tokens_per_doc

        return await selected_rerank_func(**request_kwargs)

    return local_rerank


async def run() -> None:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    config = CONFIG
    embedding_config = config.embedding
    embedding_func_impl = partial(
        openai_embed.func,
        model=embedding_config.model,
        base_url=embedding_config.base_url,
        api_key=embedding_config.api_key,
        client_configs={"timeout": embedding_config.timeout},
    )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_config.embedding_dim,
        max_token_size=embedding_config.max_token_size,
        model_name=embedding_config.model,
        func=embedding_func_impl,
    )
    rerank_model_func = build_rerank_model_func(config)

    rag = LightRAG(
        working_dir=config.working_dir,
        llm_model_name=config.llm.model,
        llm_model_func=local_llm_complete,
        embedding_func=embedding_func,
        rerank_model_func=rerank_model_func,
        min_rerank_score=config.rerank.min_score,
    )

    await rag.initialize_storages()
    try:
        custom_chunk_payloads = load_custom_chunk_payloads(config)
        if not custom_chunk_payloads:
            raise ValueError(
                "CONFIG.input_path must point to a valid custom chunk JSON/JSONL payload file"
            )

        payload_count, doc_count, chunk_count = summarize_custom_chunk_payloads(
            custom_chunk_payloads
        )
        print("\n=== Custom Chunk Import Preview ===")
        print(f"Input file: {config.input_path}")
        print(f"Payloads recognized: {payload_count}")
        print(f"Document IDs recognized: {doc_count}")
        print(f"Chunks recognized: {chunk_count}")

        for payload in custom_chunk_payloads:
            await rag.ainsert_custom_chunks(payload)

        result = await rag.aquery_llm(
            config.question,
            param=QueryParam(
                mode=config.mode,
                enable_rerank=config.rerank.enabled,
            ),
        )

        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print("\n=== LightRAG Query Result ===")
        print(f"Saved to: {output_path}")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(run())
