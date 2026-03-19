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
WORKING_DIR = Path(__file__).resolve().parent / "rag_storage" / "2wikimqa"
OUTPUT_PATH = Path(__file__).resolve().parent / "3.19" / "query_result"

WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class OpenAICompatibleLLMConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    timeout: int = 3600
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
    timeout: int = 3600


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = True
    binding: str = "cohere"
    model: str = "BAAI/bge-reranker-v2-m3"
    base_url: str = "http://127.0.0.1:8002/rerank"
    api_key: str = "EMPTY"
    min_score: float = 0.0
    enable_chunking: bool = False
    max_tokens_per_doc: int = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkippedChunkInfo:
    payload_index: int
    split_index: int
    doc_id: str | None
    chunk_id: str | None
    reason: str


@dataclass(frozen=True)
class RunConfig:
    chunk_input_path: str | None = "/data/h50056789/Rag_Chunking/test_database/3.9/2wikimqa_lumber_chunk_Qwen2.5-7B-Instruct.jsonl"
    query_input_path: str | None = None
    question: str = "Who is George V?"
    mode: str = "hybrid"
    working_dir: str = str(WORKING_DIR)
    output_path: str = str(OUTPUT_PATH)
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
            max_token_size=480,
        )
    )
    rerank: RerankConfig = field(default_factory=RerankConfig)


CONFIG = RunConfig()


def is_custom_chunk_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and "splits" in payload


def load_json_items(input_path: Path) -> list[Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        items: list[Any] = []
        with input_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        return [payload]

    raise ValueError(f"Unsupported input file type: {input_path.suffix}")


def load_custom_chunk_payloads(config: RunConfig) -> list[dict[str, Any]]:
    chunk_input = config.chunk_input_path
    if not chunk_input:
        return []

    input_path = Path(chunk_input)
    items = load_json_items(input_path)
    if not items:
        return []

    payloads: list[dict[str, Any]] = []
    for line_number, item in enumerate(items, start=1):
        if not is_custom_chunk_payload(item):
            raise ValueError(
                f"Input item {line_number} is not a custom chunk payload with 'splits'"
            )
        payloads.append(item)

    return payloads


def is_query_record(payload: Any) -> bool:
    return isinstance(payload, dict) and any(
        key in payload for key in ("user_input", "question", "query")
    )


def normalize_query_record(record: dict[str, Any]) -> dict[str, Any]:
    question = (
        record.get("user_input")
        or record.get("question")
        or record.get("query")
    )
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Query record must contain a non-empty user_input/question/query")

    reference = record.get("reference")
    if reference is None:
        reference = record.get("ground_truth")

    meta = record.get("meta")
    if meta is None:
        meta = {}

    return {
        "user_input": question.strip(),
        "reference": reference,
        "meta": meta,
    }


def load_query_records(config: RunConfig) -> list[dict[str, Any]]:
    query_input = config.query_input_path

    if not query_input:
        return [
            {
                "user_input": config.question,
                "reference": None,
                "meta": {},
            }
        ]

    input_path = Path(query_input)
    items = load_json_items(input_path)
    if not items:
        raise ValueError(f"No query records found in {input_path}")

    records: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("test_cases"), list):
            for nested_item in item["test_cases"]:
                if not isinstance(nested_item, dict):
                    raise ValueError("Each test_cases item must be a JSON object")
                records.append(normalize_query_record(nested_item))
            continue

        if not isinstance(item, dict) or not is_query_record(item):
            raise ValueError(
                "Query input must be a query record, a JSONL of query records, "
                "or a JSON object containing a test_cases array"
            )
        records.append(normalize_query_record(item))

    return records


def extract_retrieval_list(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    data = query_result.get("data", {})
    chunks = data.get("chunks", [])
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    return []


def build_output_record(
    query_record: dict[str, Any],
    query_result: dict[str, Any],
) -> dict[str, Any]:
    llm_response = query_result.get("llm_response", {})
    llm_answer = None
    if isinstance(llm_response, dict):
        llm_answer = llm_response.get("content")

    return {
        "user_input": query_record["user_input"],
        "reference": query_record.get("reference"),
        "meta": query_record.get("meta", {}),
        "llm_answer": llm_answer,
        "retrieval_list": extract_retrieval_list(query_result),
    }


def extract_chunk_doc_id(raw_chunk: Any, default_doc_id: str | None = None) -> str | None:
    if isinstance(raw_chunk, dict):
        return raw_chunk.get("doc_id") or raw_chunk.get("full_doc_id") or default_doc_id
    if isinstance(raw_chunk, (list, tuple)):
        if len(raw_chunk) > 1 and raw_chunk[1] not in (None, ""):
            return str(raw_chunk[1])
        return default_doc_id
    return default_doc_id


def extract_chunk_id(raw_chunk: Any) -> str | None:
    if isinstance(raw_chunk, dict):
        chunk_id = raw_chunk.get("chunk_id", raw_chunk.get("id"))
        if chunk_id not in (None, ""):
            return str(chunk_id)
        return None
    if isinstance(raw_chunk, (list, tuple)):
        if len(raw_chunk) > 2 and raw_chunk[2] not in (None, ""):
            return str(raw_chunk[2])
        return None
    return None


def filter_invalid_custom_chunk_payloads(
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[SkippedChunkInfo]]:
    from lightrag.lightrag import _normalize_custom_chunk_entry

    filtered_payloads: list[dict[str, Any]] = []
    skipped_chunks: list[SkippedChunkInfo] = []

    for payload_index, payload in enumerate(payloads, start=1):
        payload_doc_id = payload.get("doc_id", payload.get("full_doc_id"))
        payload_file_path = payload.get(
            "file_path", payload.get("filepath", "unknown_source")
        )
        splits = payload.get("splits", [])
        valid_splits: list[Any] = []

        for split_index, raw_chunk in enumerate(splits, start=1):
            doc_id = extract_chunk_doc_id(raw_chunk, payload_doc_id)
            chunk_id = extract_chunk_id(raw_chunk)

            try:
                _normalize_custom_chunk_entry(
                    raw_chunk=raw_chunk,
                    index=split_index - 1,
                    default_doc_id=payload_doc_id,
                    default_file_path=payload_file_path,
                )
            except (TypeError, ValueError) as exc:
                skipped_chunks.append(
                    SkippedChunkInfo(
                        payload_index=payload_index,
                        split_index=split_index,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        reason=str(exc),
                    )
                )
                continue

            valid_splits.append(raw_chunk)

        if valid_splits:
            filtered_payload = dict(payload)
            filtered_payload["splits"] = valid_splits
            filtered_payloads.append(filtered_payload)

    return filtered_payloads, skipped_chunks


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
        max_parallel_insert=10,
        default_llm_timeout=config.llm.timeout,
    )

    await rag.initialize_storages()
    try:
        custom_chunk_payloads = load_custom_chunk_payloads(config)
        custom_chunk_payloads, skipped_chunks = filter_invalid_custom_chunk_payloads(
            custom_chunk_payloads
        )
        if custom_chunk_payloads:
            payload_count, doc_count, chunk_count = summarize_custom_chunk_payloads(
                custom_chunk_payloads
            )
            print("\n=== Custom Chunk Import Preview ===")
            print(f"Input file: {config.chunk_input_path}")
            print(f"Payloads: {payload_count}  Documents: {doc_count}  "
                  f"Chunks: {chunk_count}  Skipped: {len(skipped_chunks)}")

            if skipped_chunks:
                for skipped in skipped_chunks[:5]:
                    print(
                        f"  [skipped] split={skipped.split_index} "
                        f"doc_id={skipped.doc_id} reason={skipped.reason}"
                    )
                if len(skipped_chunks) > 5:
                    print(f"  ... and {len(skipped_chunks) - 5} more")

            for payload in custom_chunk_payloads:
                await rag.ainsert_custom_chunks(payload)

        query_records = load_query_records(config)
        results: list[dict[str, Any]] = []

        print("\n=== Query Execution Preview ===")
        print(f"Query count: {len(query_records)}")
        print(f"Mode: {config.mode}")
        if config.query_input_path:
            print(
                "Question source: query_input_path "
                f"({config.query_input_path}); question is ignored when a query file is set"
            )
        else:
            print("Question source: question")

        for index, query_record in enumerate(query_records, start=1):
            print(f"Running query {index}/{len(query_records)}")
            result = await rag.aquery_llm(
                query_record["user_input"],
                param=QueryParam(
                    mode=config.mode,
                    enable_rerank=config.rerank.enabled,
                ),
            )
            # {
            # "status": "success",
            # "message": "Query processed successfully",
            # "data": {
            #     "entities": [...],
            #     "relationships": [...],
            #     "chunks": [...],
            #     "references": [...]
            # },
            # "metadata": {
            #     "query_mode": "hybrid",
            #     "keywords": {
            #     "high_level": [...],
            #     "low_level": [...]
            #     },
            #     "processing_info": {
            #     "...": "..."
            #     }
            # },
            # "llm_response": {
            #     "content": "LLM最终回答",
            #     "response_iterator": null,
            #     "is_streaming": false
            # }
            # }
            results.append(build_output_record(query_record, result))

        output_path = Path(config.output_path)
        output_payload: dict[str, Any] | list[dict[str, Any]]
        output_payload = results[0] if len(results) == 1 else results
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        print("\n=== LightRAG Query Result ===")
        print(f"Saved to: {output_path}")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(run())
