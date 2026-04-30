import asyncio
import json
import sys
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from run_lightrag import (
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleLLMConfig,
    _peek_logging_config,
    _resolve_jobs_file,
    bge_openai_compatible_embed,
    build_local_llm_complete,
    load_json_items,
    setup_runner_logging,
)
from milvus_lite_config import (
    MilvusLiteConfig,
    build_milvus_vector_storage_kwargs,
    configure_local_milvus_lite,
)

WORKING_DIR = Path(__file__).resolve().parent / "rag_storage_milvus" / "LightRAG"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "default_query.json"


@dataclass(frozen=True)
class QueryDefaults:
    """Per-job defaults forwarded to QueryParam. Leave a field as None to let
    LightRAG / QueryParam use its own default (usually env-var driven)."""

    top_k: int | None = None
    chunk_top_k: int | None = None
    max_entity_tokens: int | None = None
    max_relation_tokens: int | None = None
    max_total_tokens: int | None = None
    user_prompt: str | None = None


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
class QueryRunConfig:
    query_input_path: str | None = None
    question: str = "Which country the director of film Renegade Force is from?"
    mode: str = "mix"
    # Concurrent query workers. Each worker emits ~2 LLM calls per query
    # (keyword extract + final answer) which share LightRAG's global
    # llm_model_max_async semaphore, so real LLM QPS is capped there.
    max_parallel_queries: int = 5
    llm_model_max_async: int = 10
    vector_storage: str = "MilvusVectorDBStorage"
    working_dir: str = str(WORKING_DIR)
    output_path: str = str(OUTPUT_PATH)
    # Default off for evaluation runs: avoid stale answers when the underlying
    # graph changes but the query string stays the same. Override per job.
    enable_llm_cache: bool = False
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
    query: QueryDefaults = field(default_factory=QueryDefaults)


CONFIG = QueryRunConfig()


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

    question_id = record.get("question_id")
    if question_id is None and isinstance(meta, dict):
        question_id = meta.get("question_id")

    return {
        "user_input": question.strip(),
        "reference": reference,
        "meta": meta,
        "question_id": question_id,
    }


def load_query_records(config: QueryRunConfig) -> list[dict[str, Any]]:
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


def _extract_list(query_result: dict[str, Any], key: str) -> list[dict[str, Any]]:
    data = query_result.get("data", {})
    items = data.get(key, [])
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def extract_retrieval_list(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_list(query_result, "chunks")


def extract_entities_list(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_list(query_result, "entities")


def extract_relationships_list(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_list(query_result, "relationships")


def extract_references_list(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_list(query_result, "references")


def extract_retrieval_metadata(query_result: dict[str, Any]) -> dict[str, Any]:
    metadata = query_result.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _normalize_reference_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _flatten_gold_reference(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten meta.reference_contexts into the shape RetrievalEvaluator expects.

    Source: `meta.reference_contexts` is a nested list of groups, each group is
    a list of {doc_id, chunk_id, source_filepath, ...}. We emit a flat list of
    {doc_id, chunk_id, filepath}. `text` is intentionally omitted — every gold
    item has both doc_id and chunk_id, so RetrievalEvaluator's strongest match
    branch (`doc:{did}#chunk:{cid}`) is used and `text` is never read.
    """
    if not isinstance(meta, dict):
        return []
    raw = meta.get("reference_contexts")
    if not isinstance(raw, list):
        return []

    flattened: list[dict[str, Any]] = []
    for group in raw:
        items = group if isinstance(group, list) else [group]
        for item in items:
            if not isinstance(item, dict):
                continue
            flattened.append(
                {
                    "doc_id": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id"),
                    "filepath": (
                        item.get("filepath")
                        or item.get("source_filepath")
                        or item.get("file_path")
                    ),
                }
            )
    return flattened


def _rename_chunks_for_retrieval_eval(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map LightRAG chunk fields onto the names RetrievalEvaluator / RAGAS read:
    `content` -> `text`, `file_path` -> `filepath`. Critically, the `chunk_id`
    we emit must be the *user-supplied* id (LightRAG stores this as
    `custom_chunk_id`), not LightRAG's internal mdhash chunk_id — the gold
    references use the user-supplied ids, so matching against the hash would
    always miss."""
    renamed: list[dict[str, Any]] = []
    for chunk in chunks:
        custom_chunk_id = chunk.get("custom_chunk_id", "")
        # Fallback to LightRAG's internal id only when no user id was supplied
        # (e.g. chunks ingested without a custom chunker). Eval will then drop
        # to the text-hash matching branch in _doc_key.
        chunk_id_for_eval = (
            custom_chunk_id if custom_chunk_id not in (None, "") else chunk.get("chunk_id", "")
        )
        renamed.append(
            {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk_id_for_eval,
                "text": chunk.get("content", ""),
                "filepath": chunk.get("file_path", ""),
                "lightrag_chunk_id": chunk.get("chunk_id", ""),
                "reference_id": chunk.get("reference_id", ""),
            }
        )
    return renamed


def build_output_record(
    query_record: dict[str, Any],
    query_result: dict[str, Any],
) -> dict[str, Any]:
    llm_response = query_result.get("llm_response", {})
    llm_answer = None
    if isinstance(llm_response, dict):
        llm_answer = llm_response.get("content")

    return {
        # Field names chosen so each evaluator's fallback chain hits exactly one
        # of these keys; no per-concept duplication.
        "_id": query_record.get("question_id"),
        "question": query_record["user_input"],
        "llm_ans": llm_answer,
        "answers": _normalize_reference_to_list(query_record.get("reference")),
        "rag_retrieval": _rename_chunks_for_retrieval_eval(
            extract_retrieval_list(query_result)
        ),
        "gold_reference": _flatten_gold_reference(query_record.get("meta") or {}),
        # Pass-through context kept for inspection / future evaluators
        "entities": extract_entities_list(query_result),
        "relationships": extract_relationships_list(query_result),
        "references": extract_references_list(query_result),
        "retrieval_metadata": extract_retrieval_metadata(query_result),
    }


def build_rerank_model_func(config: QueryRunConfig):
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


def _strip_comments(overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    return {k: v for k, v in overrides.items() if not k.startswith("_")}


def _merge_dataclass(base: Any, overrides: dict[str, Any] | None) -> Any:
    effective = _strip_comments(overrides)
    if not effective:
        return base
    allowed = {f.name for f in base.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(effective) - allowed
    if unknown:
        raise ValueError(
            f"Unknown override keys for {type(base).__name__}: {sorted(unknown)}"
        )
    return replace(base, **effective)


def apply_job_overrides(base: QueryRunConfig, job: dict[str, Any]) -> QueryRunConfig:
    top_level_keys = {
        "query_input_path",
        "question",
        "mode",
        "max_parallel_queries",
        "llm_model_max_async",
        "vector_storage",
        "working_dir",
        "output_path",
        "enable_llm_cache",
    }
    effective_job = {k: v for k, v in job.items() if not k.startswith("_")}
    unknown = set(effective_job) - (
        top_level_keys
        | {"llm", "embedding", "rerank", "milvus", "query", "name", "enabled"}
    )
    if unknown:
        raise ValueError(
            f"Unknown job keys: {sorted(unknown)}. "
            "Build-only fields (chunk_input_path, max_parallel_insert) are not "
            "accepted here; keep them in the build jobs file instead."
        )

    top_overrides = {k: effective_job[k] for k in top_level_keys if k in effective_job}
    llm = _merge_dataclass(base.llm, effective_job.get("llm"))
    embedding = _merge_dataclass(base.embedding, effective_job.get("embedding"))
    rerank = _merge_dataclass(base.rerank, effective_job.get("rerank"))
    query = _merge_dataclass(base.query, effective_job.get("query"))

    milvus_override = effective_job.get("milvus")
    effective_milvus_override = _strip_comments(milvus_override)
    if "working_dir" in effective_job and "uri" not in effective_milvus_override:
        milvus = replace(
            _merge_dataclass(base.milvus, milvus_override),
            uri=None,
        )
    else:
        milvus = _merge_dataclass(base.milvus, milvus_override)

    return replace(
        base,
        **top_overrides,
        llm=llm,
        embedding=embedding,
        rerank=rerank,
        milvus=milvus,
        query=query,
    )


def load_jobs_file(
    jobs_path: Path, base: QueryRunConfig
) -> list[tuple[str, QueryRunConfig]]:
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path}")

    raw = json.loads(jobs_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Jobs file must be a JSON object with 'jobs' array")

    defaults = raw.get("defaults") or {}
    base_with_defaults = apply_job_overrides(base, defaults) if defaults else base

    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("Jobs file must contain a non-empty 'jobs' array")

    resolved: list[tuple[str, QueryRunConfig]] = []
    for index, job in enumerate(jobs_raw, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"Job #{index} must be a JSON object")
        if job.get("enabled", True) is False:
            continue
        name = str(job.get("name") or f"job_{index}")
        resolved.append((name, apply_job_overrides(base_with_defaults, job)))

    if not resolved:
        raise ValueError("All jobs are disabled; nothing to run")
    return resolved


def build_query_param_kwargs(config: QueryRunConfig) -> dict[str, Any]:
    query_param_kwargs: dict[str, Any] = {
        "mode": config.mode,
        "enable_rerank": config.rerank.enabled,
    }
    # Only forward fields explicitly set by the job/config. If left as None,
    # QueryParam keeps its own env-var-driven defaults.
    for field_name in (
        "top_k",
        "chunk_top_k",
        "max_entity_tokens",
        "max_relation_tokens",
        "max_total_tokens",
        "user_prompt",
    ):
        value = getattr(config.query, field_name)
        if value is not None:
            query_param_kwargs[field_name] = value
    return query_param_kwargs


def build_query_rag(config: QueryRunConfig) -> tuple[LightRAG, str | None]:
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

    rag = LightRAG(
        working_dir=config.working_dir,
        llm_model_name=config.llm.model,
        llm_model_func=build_local_llm_complete(config.llm),
        embedding_func=embedding_func,
        rerank_model_func=build_rerank_model_func(config),
        min_rerank_score=config.rerank.min_score,
        llm_model_max_async=max(1, config.llm_model_max_async),
        default_llm_timeout=config.llm.timeout,
        vector_storage=config.vector_storage,
        vector_db_storage_cls_kwargs=build_milvus_vector_storage_kwargs(
            config.milvus
        ),
        enable_llm_cache=config.enable_llm_cache,
    )
    return rag, milvus_db_path


async def run_query_records(
    rag: LightRAG,
    query_records: list[dict[str, Any]],
    *,
    max_parallel_queries: int,
    query_param_factory: Callable[[], QueryParam],
) -> tuple[list[dict[str, Any] | None], list[tuple[int, dict[str, Any], Exception]]]:
    """Run queries concurrently with per-query error isolation.

    Returns a tuple of (results_by_index, errors). results_by_index preserves
    the original order and contains None for failed queries so callers can
    still cross-reference against the input list. errors is a list of
    (1-based-index, query_record, exception).
    """
    total = len(query_records)
    max_parallel_queries = max(1, max_parallel_queries)
    worker_count = min(max_parallel_queries, total)

    print("\n=== Query Execution ===")
    print(f"Pending queries: {total}")
    print(f"Parallel query workers: {worker_count}")

    if total == 0:
        return [], []

    queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
    for index, query_record in enumerate(query_records, start=1):
        queue.put_nowait((index, query_record))
    for _ in range(worker_count):
        queue.put_nowait(None)

    results: list[dict[str, Any] | None] = [None] * total
    progress_lock = asyncio.Lock()
    error_lock = asyncio.Lock()
    completed = 0
    failed = 0
    query_errors: list[tuple[int, dict[str, Any], Exception]] = []

    async def worker(worker_index: int) -> None:
        nonlocal completed, failed

        while True:
            item = await queue.get()
            try:
                if item is None:
                    return

                index, query_record = item
                print(
                    f"[worker {worker_index}] Running query {index}/{total}: "
                    f"{query_record['user_input']}"
                )
                try:
                    result = await rag.aquery_llm(
                        query_record["user_input"],
                        param=query_param_factory(),
                    )
                    results[index - 1] = build_output_record(query_record, result)
                    async with progress_lock:
                        completed += 1
                        print(
                            f"[worker {worker_index}] Completed query {index}/{total} "
                            f"(ok={completed} failed={failed})"
                        )
                except Exception as exc:
                    # Isolate per-query failures so the remaining queries still
                    # run. Typical causes: transient LLM timeout, a single
                    # malformed input. We capture the full exception for the
                    # final summary written alongside the results file.
                    async with error_lock:
                        query_errors.append((index, query_record, exc))
                    async with progress_lock:
                        failed += 1
                        print(
                            f"[worker {worker_index}] FAILED query {index}/{total}: "
                            f"{exc!r} (ok={completed} failed={failed})"
                        )
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker(worker_index))
        for worker_index in range(1, worker_count + 1)
    ]

    await queue.join()
    await asyncio.gather(*workers, return_exceptions=True)

    return results, query_errors


def _assert_working_dir_has_built_graph(config: QueryRunConfig) -> None:
    """Abort the query job if `working_dir` does not look like a built LightRAG
    store. LightRAG silently creates an empty working_dir / Milvus collection
    when it cannot find one, which would yield 0-recall evaluation runs that
    look like model failure rather than a misconfigured path.
    """
    working_dir = Path(config.working_dir)
    if not working_dir.is_dir():
        raise FileNotFoundError(
            f"working_dir does not exist: {working_dir}. "
            "Build the index for this job first, or fix the path."
        )

    graph_file = working_dir / "graph_chunk_entity_relation.graphml"
    milvus_db = working_dir / "milvus_lite.db"

    missing: list[str] = []
    if not graph_file.is_file() or graph_file.stat().st_size == 0:
        missing.append(str(graph_file))
    if not milvus_db.is_file() or milvus_db.stat().st_size == 0:
        missing.append(str(milvus_db))

    if missing:
        raise FileNotFoundError(
            "working_dir is missing built artifacts (file absent or empty): "
            + ", ".join(missing)
            + ". This job would query an empty knowledge graph; aborting to "
            "avoid silently producing 0-recall evaluation results."
        )


def _warn_on_retrieval_mode_mismatch(config: QueryRunConfig) -> None:
    """Flag query-time Milvus settings that do not match common build outputs.

    These are not hard errors (LightRAG / Milvus will surface the real error
    at query time) but catching them early helps the user diagnose an empty
    retrieval_list faster than reading Milvus's own error path.
    """
    m = config.milvus
    if m.enable_hybrid_search and not m.enable_sparse:
        print(
            "!!! WARNING: milvus.enable_hybrid_search=true requires "
            "milvus.enable_sparse=true. Milvus will reject the hybrid query."
        )
    if m.enable_hybrid_search:
        print(
            "[info] hybrid search enabled; make sure this working_dir was "
            "built with milvus.enable_sparse=true, otherwise queries return empty."
        )


async def run_query_job(name: str, config: QueryRunConfig) -> None:
    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n========== Query Job: {name} ==========")
    print(f"working_dir={config.working_dir}")
    print(f"query_input_path={config.query_input_path}")
    print(f"output_path={config.output_path}")
    print(f"mode={config.mode}")
    print(
        f"milvus.enable_hybrid_search={config.milvus.enable_hybrid_search} "
        f"milvus.enable_sparse={config.milvus.enable_sparse}"
    )
    print(f"rerank.enabled={config.rerank.enabled}")
    print(f"enable_llm_cache={config.enable_llm_cache}")
    _warn_on_retrieval_mode_mismatch(config)
    _assert_working_dir_has_built_graph(config)

    rag, milvus_db_path = build_query_rag(config)
    try:
        # initialize_storages inside the try block so that a partial init
        # failure still runs finalize_storages on whatever was opened.
        await rag.initialize_storages()

        query_records = load_query_records(config)
        query_param_kwargs = build_query_param_kwargs(config)

        print("\n=== Query Execution Preview ===")
        print(f"Query count: {len(query_records)}")
        print(f"Mode: {config.mode}")
        print(f"Working dir: {config.working_dir}")
        print(f"Milvus Lite DB: {milvus_db_path}")
        if config.query_input_path:
            print(
                "Question source: query_input_path "
                f"({config.query_input_path}); question is ignored when a query file is set"
            )
        else:
            print("Question source: question")
        print(
            "QueryParam overrides: "
            + ", ".join(
                f"{k}={v}" for k, v in query_param_kwargs.items() if k != "mode"
            )
        )

        results, query_errors = await run_query_records(
            rag,
            query_records,
            max_parallel_queries=config.max_parallel_queries,
            query_param_factory=lambda: QueryParam(**query_param_kwargs),
        )

        success_count = sum(1 for r in results if r is not None)
        total = len(query_records)
        print("\n=== Query Summary ===")
        print(f"Success: {success_count}/{total}  Failed: {len(query_errors)}/{total}")
        for err_index, err_record, err_exc in query_errors[:10]:
            print(f"  [#{err_index}] {err_record['user_input'][:60]!r}: {err_exc!r}")
        if len(query_errors) > 10:
            print(f"  ... and {len(query_errors) - 10} more")

        # Build payload: always a list for batches; keep single-record dict
        # for the one-query case so existing downstream consumers still work.
        successful_records = [r for r in results if r is not None]
        if total == 1 and not query_errors:
            output_payload: dict[str, Any] | list[dict[str, Any]] = (
                successful_records[0] if successful_records else {}
            )
        else:
            output_payload = successful_records

        output_path = Path(config.output_path)
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Saved results to: {output_path}")

        if query_errors:
            errors_path = output_path.with_suffix(output_path.suffix + ".errors.json")
            errors_path.write_text(
                json.dumps(
                    [
                        {
                            "index": idx,
                            "user_input": rec.get("user_input"),
                            "reference": rec.get("reference"),
                            "meta": rec.get("meta", {}),
                            "error": repr(exc),
                        }
                        for idx, rec, exc in query_errors
                    ],
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            print(f"Saved error summary to: {errors_path}")
    finally:
        await rag.finalize_storages()


async def run() -> None:
    jobs_file = _resolve_jobs_file()
    log_overrides = _peek_logging_config(jobs_file)
    # Default the log name prefix to 'query' so query runs don't overwrite
    # build logs sharing the same directory.
    setup_runner_logging(
        log_dir=log_overrides.get("log_dir"),
        name=log_overrides.get("name", "query"),
        level=log_overrides.get("level"),
    )

    if jobs_file is None:
        await run_query_job("default", CONFIG)
        return

    jobs = load_jobs_file(jobs_file, CONFIG)
    print(f"Loaded {len(jobs)} query job(s) from {jobs_file}")
    for index, (name, config) in enumerate(jobs, start=1):
        print(f"\n>>> [{index}/{len(jobs)}] starting query job: {name}")
        try:
            await run_query_job(name, config)
        except Exception as exc:
            print(f"!!! query job '{name}' failed: {exc}")
            import traceback

            traceback.print_exc()
    print("\n========== All query jobs finished ==========")


if __name__ == "__main__":
    asyncio.run(run())
