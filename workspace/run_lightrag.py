import asyncio
import base64
import json
import sys
import os
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import Any
import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from milvus_lite_config import (
    MilvusLiteConfig,
    build_milvus_vector_storage_kwargs,
    configure_local_milvus_lite,
)

os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056789/Rag_Chunking/tiktoken_cache"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKING_DIR = Path(__file__).resolve().parent / "rag_storage_milvus" / "LightRAG"
OUTPUT_PATH = Path(__file__).resolve().parent / "3.19" / "test.json"

load_dotenv(PROJECT_ROOT / ".env", override=False)

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
    temperature: float | None = 0.0
    max_tokens: int | None = 8192
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
    max_tokens_per_doc: int = 8192
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkippedChunkInfo:
    payload_index: int
    split_index: int
    doc_id: str | None
    chunk_id: str | None
    reason: str


@dataclass
class CustomChunkInsertProgress:
    total_documents: int
    max_parallel_insert: int
    llm_model_max_async: int
    started_documents: int = 0
    completed_documents: int = 0
    failed_documents: int = 0
    active_documents: dict[str, str] = field(default_factory=dict)

    @property
    def finished_documents(self) -> int:
        return self.completed_documents + self.failed_documents

    def render_bar(self, width: int = 28) -> str:
        if self.total_documents <= 0:
            return "[" + ("#" * width) + "]"

        progress_ratio = min(1.0, self.finished_documents / self.total_documents)
        filled = int(progress_ratio * width)
        if self.finished_documents > 0 and filled == 0:
            filled = 1
        return f"[{'#' * filled}{'-' * (width - filled)}]"

    def render_summary(self) -> str:
        return (
            f"{self.render_bar()} finished={self.finished_documents}/{self.total_documents} "
            f"completed={self.completed_documents} failed={self.failed_documents} "
            f"active={len(self.active_documents)}/{self.max_parallel_insert} "
            f"MAX_PARALLEL_INSERT={self.max_parallel_insert} "
            f"MAX_ASYNC={self.llm_model_max_async}"
        )

    def render_active_documents(self, limit: int = 5) -> str | None:
        if not self.active_documents:
            return None

        active_items = list(self.active_documents.items())[:limit]
        active_text = ", ".join(
            f"{position}:{doc_id}" for doc_id, position in active_items
        )
        remaining = len(self.active_documents) - len(active_items)
        if remaining > 0:
            active_text = f"{active_text}, ... (+{remaining} more)"
        return f"Active documents: {active_text}"


@dataclass(frozen=True)
class RunConfig:
    chunk_input_path: str | None = "/data/h50056789/Rag_Chunking/Chunk_Result/Lightrag_Chunk/2wikimqa_lightrag_chunk.json"
    query_input_path: str | None = None
    question: str = "Which country the director of film Renegade Force is from?"
    mode: str = "mix"
    max_parallel_insert: int = field(
        default_factory=lambda: int(os.getenv("MAX_PARALLEL_INSERT", "10"))
    )
    llm_model_max_async: int = field(
        default_factory=lambda: int(os.getenv("MAX_ASYNC", "10"))
    )
    vector_storage: str = "MilvusVectorDBStorage"
    working_dir: str = str(WORKING_DIR)
    output_path: str = str(OUTPUT_PATH)
    llm: OpenAICompatibleLLMConfig = field(
        default_factory=lambda: OpenAICompatibleLLMConfig(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            base_url="http://127.0.0.1:8001/v1",
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=8192,
            extra_body={},
        )
    )
    embedding: OpenAICompatibleEmbeddingConfig = field(
        default_factory=lambda: OpenAICompatibleEmbeddingConfig(
            model="BAAI/bge-m3",
            base_url="http://127.0.0.1:8003/v1",
            api_key="EMPTY",
            embedding_dim=1024,
            max_token_size=8192,
        )
    )
    rerank: RerankConfig = field(default_factory=RerankConfig)
    milvus: MilvusLiteConfig = field(default_factory=MilvusLiteConfig)


CONFIG = RunConfig()


def is_custom_chunk_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and "splits" in payload


def load_json_items(input_path: Path) -> list[Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        text = input_path.read_text(encoding="utf-8")
        items: list[Any] = []
        decoder = json.JSONDecoder(strict=False)
        position = 0
        text_length = len(text)

        while position < text_length:
            while position < text_length and text[position].isspace():
                position += 1
            if position >= text_length:
                break

            try:
                item, next_position = decoder.raw_decode(text, position)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Failed to parse JSONL input. This loader supports either "
                    "standard one-JSON-object-per-line JSONL or multiple JSON "
                    "objects separated by whitespace. If text fields contain "
                    "literal newlines, they must be escaped as \\n inside JSON "
                    "strings, or the file should be converted to a regular .json file."
                ) from exc

            items.append(item)
            position = next_position

        return items

    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"), strict=False)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]

        raise ValueError(
            "Input JSON must contain either a payload object or a top-level array of payload objects"
        )

    raise ValueError(
        f"Unsupported input file type: {input_path.suffix}. Only .json and .jsonl are supported."
    )


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


def expand_custom_chunk_payloads_by_doc_id(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from lightrag.lightrag import _split_custom_chunks_payload_by_doc_id

    expanded_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        grouped_payloads = _split_custom_chunks_payload_by_doc_id(
            full_text=payload,
            text_chunks=None,
            doc_id=None,
            file_path=None,
        )
        if grouped_payloads:
            expanded_payloads.extend(grouped_payloads)
        else:
            expanded_payloads.append(payload)

    return expanded_payloads


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


def resolve_custom_chunk_payload_identity(
    payload: dict[str, Any],
) -> tuple[str, str]:
    from lightrag.lightrag import _normalize_custom_chunks_payload

    _, doc_key, normalized_file_path, _ = _normalize_custom_chunks_payload(
        full_text=payload,
        text_chunks=None,
        doc_id=None,
        file_path=None,
    )
    return doc_key, normalized_file_path


def group_custom_chunk_payloads_by_doc_id(
    payloads: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[tuple[str, str, int]]]:
    grouped_payloads: dict[str, list[dict[str, Any]]] = {}
    ordered_doc_keys: list[str] = []
    doc_file_paths: dict[str, str] = {}

    for payload in payloads:
        doc_key, normalized_file_path = resolve_custom_chunk_payload_identity(payload)
        if doc_key not in grouped_payloads:
            grouped_payloads[doc_key] = []
            ordered_doc_keys.append(doc_key)
            doc_file_paths[doc_key] = normalized_file_path
        grouped_payloads[doc_key].append(payload)

    ordered_groups = [grouped_payloads[doc_key] for doc_key in ordered_doc_keys]
    duplicate_doc_groups = [
        (doc_key, doc_file_paths[doc_key], len(grouped_payloads[doc_key]))
        for doc_key in ordered_doc_keys
        if len(grouped_payloads[doc_key]) > 1
    ]
    return ordered_groups, duplicate_doc_groups


async def insert_custom_chunk_payloads(
    rag: Any,
    payloads: list[dict[str, Any]],
    *,
    max_parallel_payloads: int,
) -> None:
    if not payloads:
        return

    payload_groups, duplicate_doc_groups = group_custom_chunk_payloads_by_doc_id(
        payloads
    )
    effective_max_parallel_insert = max(
        1, int(getattr(rag, "max_parallel_insert", max_parallel_payloads))
    )
    effective_llm_model_max_async = max(
        1, int(getattr(rag, "llm_model_max_async", 1))
    )
    worker_count = min(effective_max_parallel_insert, len(payload_groups))
    progress = CustomChunkInsertProgress(
        total_documents=len(payload_groups),
        max_parallel_insert=effective_max_parallel_insert,
        llm_model_max_async=effective_llm_model_max_async,
    )
    progress_lock = asyncio.Lock()

    print("\n=== Custom Chunk Insert Execution ===")
    print(f"Pending payloads: {len(payloads)}")
    print(f"Document groups: {len(payload_groups)}")
    print(f"Parallel document workers: {worker_count}")
    print(f"Effective MAX_PARALLEL_INSERT: {effective_max_parallel_insert}")
    print(f"Effective MAX_ASYNC: {effective_llm_model_max_async}")
    print(
        "Concurrency alignment: runner controls document-level parallelism; "
        "LightRAG keeps per-document extraction and graph-merge parallelism."
    )
    print(f"Progress {progress.render_summary()}")

    if duplicate_doc_groups:
        print(
            "Repeated doc_ids detected; those payloads will stay sequential within the same document."
        )
        for doc_key, file_path, count in duplicate_doc_groups[:10]:
            print(f"  doc_id={doc_key} file_path={file_path} payloads={count}")
        if len(duplicate_doc_groups) > 10:
            print(f"  ... and {len(duplicate_doc_groups) - 10} more")

    queue: asyncio.Queue[tuple[int, list[dict[str, Any]]] | None] = asyncio.Queue()
    for group_index, payload_group in enumerate(payload_groups, start=1):
        queue.put_nowait((group_index, payload_group))
    for _ in range(worker_count):
        queue.put_nowait(None)

    insert_errors: list[Exception] = []
    error_lock = asyncio.Lock()

    async def print_progress_event(
        *,
        event: str,
        worker_index: int,
        group_index: int,
        doc_key: str,
        normalized_file_path: str,
        payload_count: int,
        error: Exception | None = None,
    ) -> None:
        async with progress_lock:
            document_position = f"{group_index}/{len(payload_groups)}"

            if event == "start":
                progress.started_documents += 1
                progress.active_documents[doc_key] = document_position
                print(
                    f"[worker {worker_index}] PROCESSING document {document_position}: "
                    f"doc_id={doc_key} payloads={payload_count} "
                    f"file_path={normalized_file_path}"
                )
            elif event == "complete":
                progress.active_documents.pop(doc_key, None)
                progress.completed_documents += 1
                print(
                    f"[worker {worker_index}] COMPLETED document {document_position}: "
                    f"doc_id={doc_key} file_path={normalized_file_path}"
                )
            elif event == "failed":
                progress.active_documents.pop(doc_key, None)
                progress.failed_documents += 1
                print(
                    f"[worker {worker_index}] FAILED document {document_position}: "
                    f"doc_id={doc_key} file_path={normalized_file_path} error={error}"
                )

            print(f"Progress {progress.render_summary()}")
            active_line = progress.render_active_documents()
            if active_line:
                print(active_line)

    async def worker(worker_index: int) -> None:
        while True:
            item = await queue.get()
            doc_started = False
            doc_key = "unknown_doc"
            normalized_file_path = "unknown_source"
            payload_group: list[dict[str, Any]] = []
            group_index = 0
            try:
                if item is None:
                    return

                group_index, payload_group = item
                doc_key, normalized_file_path = resolve_custom_chunk_payload_identity(
                    payload_group[0]
                )
                await print_progress_event(
                    event="start",
                    worker_index=worker_index,
                    group_index=group_index,
                    doc_key=doc_key,
                    normalized_file_path=normalized_file_path,
                    payload_count=len(payload_group),
                )
                doc_started = True

                for payload in payload_group:
                    await rag.ainsert_custom_chunks(
                        payload,
                        current_file_number=group_index,
                        total_files=len(payload_groups),
                    )

                await print_progress_event(
                    event="complete",
                    worker_index=worker_index,
                    group_index=group_index,
                    doc_key=doc_key,
                    normalized_file_path=normalized_file_path,
                    payload_count=len(payload_group),
                )
            except Exception as exc:
                if doc_started:
                    await print_progress_event(
                        event="failed",
                        worker_index=worker_index,
                        group_index=group_index,
                        doc_key=doc_key,
                        normalized_file_path=normalized_file_path,
                        payload_count=len(payload_group),
                        error=exc,
                    )
                async with error_lock:
                    insert_errors.append(exc)
                # Continue processing remaining documents instead of stopping
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker(worker_index))
        for worker_index in range(1, worker_count + 1)
    ]

    await queue.join()
    await asyncio.gather(*workers, return_exceptions=True)

    print("\n=== Custom Chunk Insert Summary ===")
    print(f"Progress {progress.render_summary()}")
    active_line = progress.render_active_documents()
    if active_line:
        print(active_line)

    if insert_errors:
        print(f"\n{len(insert_errors)} document(s) failed during insertion:")
        for i, err in enumerate(insert_errors[:10], 1):
            print(f"  [{i}] {err}")
        if len(insert_errors) > 10:
            print(f"  ... and {len(insert_errors) - 10} more")


@lru_cache(maxsize=4)
def get_embedding_hf_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            f"Tokenizer for {model_name} must be a fast tokenizer to provide offset mappings."
        )
    return tokenizer


def count_text_tokens(
    content: str,
    tokenizer: Any,
) -> int:
    return len(
        tokenizer(
            content,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )


def collect_custom_chunk_token_counts(
    payloads: list[dict[str, Any]],
    embedding_config: OpenAICompatibleEmbeddingConfig,
) -> list[dict[str, Any]]:
    from lightrag.lightrag import _normalize_custom_chunk_entry

    tokenizer = get_embedding_hf_tokenizer(embedding_config.model)
    token_rows: list[dict[str, Any]] = []

    for payload_index, payload in enumerate(payloads, start=1):
        payload_doc_id = payload.get("doc_id", payload.get("full_doc_id"))
        payload_file_path = payload.get(
            "file_path", payload.get("filepath", "unknown_source")
        )

        for split_index, raw_chunk in enumerate(payload.get("splits", []), start=1):
            normalized_chunk = _normalize_custom_chunk_entry(
                raw_chunk=raw_chunk,
                index=split_index - 1,
                default_doc_id=payload_doc_id,
                default_file_path=payload_file_path,
            )
            token_count = count_text_tokens(normalized_chunk["content"], tokenizer)
            token_rows.append(
                {
                    "payload_index": payload_index,
                    "split_index": split_index,
                    "doc_id": normalized_chunk["doc_id"],
                    "chunk_id": normalized_chunk["chunk_id"],
                    "chunk_order_index": normalized_chunk["chunk_order_index"],
                    "token_count": token_count,
                    "over_limit": (
                        embedding_config.max_token_size is not None
                        and embedding_config.max_token_size > 0
                        and token_count > embedding_config.max_token_size
                    ),
                }
            )

    return token_rows


async def split_existing_custom_chunk_payloads(
    rag: Any,
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from lightrag.base import DocStatus
    from lightrag.lightrag import _normalize_custom_chunks_payload

    pending_payloads: list[dict[str, Any]] = []
    existing_payloads: list[dict[str, Any]] = []

    for payload_index, payload in enumerate(payloads, start=1):
        _, doc_key, normalized_file_path, _ = _normalize_custom_chunks_payload(
            full_text=payload,
            text_chunks=None,
            doc_id=None,
            file_path=None,
        )
        existing_doc_status = await rag.doc_status.get_by_id(doc_key)
        existing_status = existing_doc_status.get("status") if existing_doc_status else None

        if existing_status in (DocStatus.PROCESSING.value, DocStatus.FAILED.value):
            pending_payloads.append(payload)
            continue

        existing_full_doc = await rag.full_docs.get_by_id(doc_key)
        if existing_doc_status or existing_full_doc:
            existing_payloads.append(
                {
                    "payload_index": payload_index,
                    "doc_id": doc_key,
                    "file_path": normalized_file_path,
                    "status": existing_status or "stored",
                    "track_id": existing_doc_status.get("track_id", "")
                    if existing_doc_status
                    else "",
                }
            )
            continue

        pending_payloads.append(payload)

    return pending_payloads, existing_payloads


def truncate_text_by_tokenizer_limit(
    content: str,
    tokenizer: Any,
    max_token_size: int | None,
) -> str:
    if max_token_size is None or max_token_size <= 0 or not content:
        return content

    encoded = tokenizer(
        content,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    offsets: list[tuple[int, int]] = list(encoded.get("offset_mapping", []))
    if len(offsets) <= max_token_size:
        return content

    end_offset = offsets[max_token_size - 1][1]
    while end_offset <= 0 and max_token_size < len(offsets):
        max_token_size += 1
        end_offset = offsets[max_token_size - 1][1]

    if end_offset <= 0:
        raise ValueError(
            "Failed to derive a valid character boundary when truncating embedding text."
        )

    return content[:end_offset]


_embedding_client: Any = None
_embedding_client_key: tuple | None = None


def _get_embedding_client(
    api_key: str,
    base_url: str,
    client_configs: dict[str, Any] | None = None,
) -> Any:
    """Reuse a single AsyncOpenAI client for embedding to avoid per-call TCP overhead."""
    from lightrag.llm.openai import create_openai_async_client

    global _embedding_client, _embedding_client_key
    key = (api_key, base_url, frozenset((client_configs or {}).items()))
    if _embedding_client is None or _embedding_client_key != key:
        _embedding_client = create_openai_async_client(
            api_key=api_key,
            base_url=base_url,
            client_configs=client_configs,
        )
        _embedding_client_key = key
    return _embedding_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def bge_openai_compatible_embed(
    texts: list[str],
    model: str,
    base_url: str,
    api_key: str,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    client_configs: dict[str, Any] | None = None,
    token_tracker: Any | None = None,
) -> np.ndarray:
    from lightrag.utils import logger

    tokenizer = get_embedding_hf_tokenizer(model)
    truncated_texts: list[str] = []
    truncation_count = 0

    for text in texts:
        truncated_text = truncate_text_by_tokenizer_limit(
            text,
            tokenizer,
            max_token_size,
        )
        if truncated_text != text:
            truncation_count += 1
        truncated_texts.append(truncated_text)

    if truncation_count > 0 and max_token_size is not None and max_token_size > 0:
        logger.info(
            "Truncated %d/%d texts with %s tokenizer to fit token limit (%d)",
            truncation_count,
            len(texts),
            model,
            max_token_size,
        )

    client = _get_embedding_client(api_key, base_url, client_configs)

    api_params = {
        "model": model,
        "input": truncated_texts,
        "encoding_format": "base64",
    }
    if embedding_dim is not None:
        api_params["dimensions"] = embedding_dim

    response = await client.embeddings.create(**api_params)

    if token_tracker and hasattr(response, "usage"):
        token_counts = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }
        token_tracker.add_usage(token_counts)

    return np.array(
        [
            np.array(dp.embedding, dtype=np.float32)
            if isinstance(dp.embedding, list)
            else np.frombuffer(
                base64.b64decode(dp.embedding), dtype=np.float32
            )
            for dp in response.data
        ]
    )

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
    from lightrag.utils import EmbeddingFunc

    config = CONFIG
    milvus_db_path = configure_local_milvus_lite(config.working_dir, config.milvus)
    embedding_config = config.embedding
    embedding_func_impl = partial(
        bge_openai_compatible_embed,
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
    max_parallel_insert = max(1, config.max_parallel_insert)
    llm_model_max_async = max(1, config.llm_model_max_async)

    rag = LightRAG(
        working_dir=config.working_dir,
        llm_model_name=config.llm.model,
        llm_model_func=local_llm_complete,
        embedding_func=embedding_func,
        rerank_model_func=rerank_model_func,
        min_rerank_score=config.rerank.min_score,
        llm_model_max_async=llm_model_max_async,
        max_parallel_insert=max_parallel_insert,
        default_llm_timeout=config.llm.timeout,
        vector_storage=config.vector_storage,
        vector_db_storage_cls_kwargs=build_milvus_vector_storage_kwargs(
            config.milvus
        ),
    )

    await rag.initialize_storages()
    try:
        print(f"Milvus Lite DB: {milvus_db_path}")
        custom_chunk_payloads = load_custom_chunk_payloads(config)
        custom_chunk_payloads = expand_custom_chunk_payloads_by_doc_id(
            custom_chunk_payloads
        )
        custom_chunk_payloads, skipped_chunks = filter_invalid_custom_chunk_payloads(
            custom_chunk_payloads
        )
        if custom_chunk_payloads:
            payload_count, doc_count, chunk_count = summarize_custom_chunk_payloads(
                custom_chunk_payloads
            )
            chunk_token_rows = collect_custom_chunk_token_counts(
                custom_chunk_payloads,
                embedding_config,
            )
            print("\n=== Custom Chunk Import Preview ===")
            print(f"Input file: {config.chunk_input_path}")
            print(f"Payloads: {payload_count}  Documents: {doc_count}  "
                  f"Chunks: {chunk_count}  Skipped: {len(skipped_chunks)}")
            print(
                f"Embedding tokenizer: {embedding_config.model}  "
                f"Limit: {embedding_config.max_token_size}"
            )
            print(f"Max parallel insert: {max_parallel_insert}")
            print(f"LLM max async: {llm_model_max_async}")

            if skipped_chunks:
                for skipped in skipped_chunks[:5]:
                    print(
                        f"  [skipped] split={skipped.split_index} "
                        f"doc_id={skipped.doc_id} reason={skipped.reason}"
                    )
                if len(skipped_chunks) > 5:
                    print(f"  ... and {len(skipped_chunks) - 5} more")

            token_average = (
                sum(row["token_count"] for row in chunk_token_rows) / len(chunk_token_rows)
                if chunk_token_rows
                else 0.0
            )
            over_limit_count = sum(1 for row in chunk_token_rows if row["over_limit"])
            print("\n=== Chunk Token Counts ===")
            print(
                f"Average tokens: {token_average:.2f}  "
                f"Over limit: {over_limit_count}/{len(chunk_token_rows)}"
            )
            for row in chunk_token_rows[:10]:
                status = "OVER_LIMIT" if row["over_limit"] else "OK"
                print(
                    f"payload={row['payload_index']} "
                    f"split={row['split_index']} "
                    f"order={row['chunk_order_index']} "
                    f"doc_id={row['doc_id']} "
                    f"chunk_id={row['chunk_id']} "
                    f"tokens={row['token_count']} "
                    f"status={status}"
                )
            if len(chunk_token_rows) > 10:
                print(f"  ... and {len(chunk_token_rows) - 10} more")

            custom_chunk_payloads, existing_payloads = await split_existing_custom_chunk_payloads(
                rag,
                custom_chunk_payloads,
            )
            if existing_payloads:
                print("\n=== Already Embedded Documents ===")
                print(f"Count: {len(existing_payloads)}")
                for row in existing_payloads[:10]:
                    print(
                        f"payload={row['payload_index']} "
                        f"doc_id={row['doc_id']} "
                        f"file_path={row['file_path']} "
                        f"status={row['status']} "
                        f"track_id={row['track_id']}"
                    )
                if len(existing_payloads) > 10:
                    print(f"  ... and {len(existing_payloads) - 10} more")

            await insert_custom_chunk_payloads(
                rag,
                custom_chunk_payloads,
                max_parallel_payloads=max_parallel_insert,
            )

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
