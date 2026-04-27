"""RAG eval harness — runs against real Pinecone + real LLM.

Usage:
    cd backend
    uv run python -m tests.evals.run_evals
    uv run python -m tests.evals.run_evals --limit 5    # smoke run

Required env vars:
    PINECONE_API_KEY, PINECONE_INDEX_NAME    — vector store
    OPENAI_API_KEY                           — embeddings (text-embedding-3-small)
    GROQ_API_KEY                             — answer generation + judge

Optional:
    EVAL_ANSWER_MODEL    — defaults to llama-3.3-70b-versatile
    EVAL_JUDGE_MODEL     — defaults to llama-3.3-70b-versatile
    PINECONE_NAMESPACE_PREFIX — defaults to ``evals`` (overrides .env to keep eval data isolated)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Force the namespace prefix BEFORE any app modules read settings, so eval data
# never lands in the prod ``rag`` namespace.
os.environ["PINECONE_NAMESPACE_PREFIX"] = os.environ.get(
    "PINECONE_NAMESPACE_PREFIX_EVAL", "evals"
)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

from langchain_openai import ChatOpenAI  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.db import Base  # noqa: E402
from app.models import ChatThread, User  # noqa: E402
from app.services.rag import (  # noqa: E402
    RagAttachment,
    RetrievalResult,
    RetrievedChunk,
    get_rag_service,
)

from tests.evals import scorers  # noqa: E402


EVALS_DIR = Path(__file__).parent
FIXTURES_DIR = EVALS_DIR / "fixtures"
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the context does not contain the information "
    "needed to answer, say explicitly that the documents do not cover it. "
    "Do not guess or invent details."
)


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        sys.stderr.write(
            f"\nMissing required environment variable: {name}\n"
            "See backend/tests/evals/README.md for setup.\n"
        )
        sys.exit(2)
    return value


def _build_groq_client(model_env: str) -> ChatOpenAI:
    api_key = _require_env("GROQ_API_KEY")
    model = (os.getenv(model_env) or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.0,
    )


def _ephemeral_session_factory():
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def _make_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:text/markdown;base64,{encoded}"


def _build_attachments() -> list[RagAttachment]:
    attachments: list[RagAttachment] = []
    for fixture in sorted(FIXTURES_DIR.glob("*.md")):
        attachments.append(
            {
                "type": "file",
                "filename": fixture.name,
                "media_type": "text/markdown",
                "url": _make_data_url(fixture),
            }
        )
    return attachments


def _format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no relevant context retrieved)"
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        label = chunk.filename or "<unnamed>"
        parts.append(f"[{index}] Source: {label}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


async def _generate_answer(
    *, question: str, retrieval: RetrievalResult, chat: ChatOpenAI
) -> str:
    context_block = _format_chunks_for_prompt(list(retrieval.chunks))
    user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}\n\nAnswer:"
    response = await chat.ainvoke(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
    )
    return (response.content or "").strip()


async def _evaluate_question(
    *,
    row: dict,
    db: Session,
    user: User,
    thread: ChatThread,
    rag_service,
    chat: ChatOpenAI,
    judge: ChatOpenAI,
) -> dict:
    question: str = row["question"]
    expected_filenames: list[str] = row.get("expected_citation_filenames") or []
    expected_keywords: list[str] = row.get("expected_keywords") or []
    must_refuse: bool = bool(row.get("must_refuse"))

    started = time.perf_counter()

    retrieval = await rag_service.retrieve(
        db=db,
        user_id=user.id,
        thread_id=thread.id,
        prompt=question,
        preferred_provider_api_key_id=None,
    )
    chunks = list(retrieval.chunks)

    answer = await _generate_answer(question=question, retrieval=retrieval, chat=chat)

    hit_rate = scorers.citation_hit_rate(chunks, expected_filenames)
    kw_coverage = scorers.keyword_coverage(answer, expected_keywords)

    precision = (
        await scorers.context_precision(
            question=question, retrieved=chunks, judge=judge
        )
        if chunks
        else 0.0
    )

    faith_score, faith_raw = await scorers.faithfulness(
        question=question, answer=answer, retrieved=chunks, judge=judge
    )
    relevance_score, relevance_raw = await scorers.answer_relevance(
        question=question, answer=answer, judge=judge
    )

    refused = scorers.is_refusal(answer)
    refusal_correct: bool | None = refused if must_refuse else None

    elapsed = time.perf_counter() - started

    return {
        "id": row["id"],
        "topic": row.get("topic", ""),
        "question": question,
        "answer": answer,
        "must_refuse": must_refuse,
        "refused": refused,
        "refusal_correct": refusal_correct,
        "retrieved_filenames": [chunk.filename for chunk in chunks],
        "retrieved_count": len(chunks),
        "scores": {
            "citation_hit_rate": round(hit_rate, 3),
            "context_precision": round(precision, 3),
            "faithfulness": round(faith_score, 3),
            "answer_relevance": round(relevance_score, 3),
            "keyword_coverage": round(kw_coverage, 3),
        },
        "judge_raw": {
            "faithfulness": faith_raw,
            "answer_relevance": relevance_raw,
        },
        "elapsed_seconds": round(elapsed, 2),
    }


_METRICS = (
    "citation_hit_rate",
    "context_precision",
    "faithfulness",
    "answer_relevance",
    "keyword_coverage",
)


def _aggregate(results: list[dict]) -> dict:
    def _avg(rows: list[dict], metric: str) -> float:
        values = [row["scores"][metric] for row in rows]
        return round(statistics.fmean(values), 3) if values else 0.0

    overall = {"count": len(results)}
    for metric in _METRICS:
        overall[metric] = _avg(results, metric)

    refusal_rows = [row for row in results if row["must_refuse"]]
    if refusal_rows:
        correct = sum(1 for row in refusal_rows if row["refusal_correct"])
        overall["refusal_accuracy"] = round(correct / len(refusal_rows), 3)
        overall["refusal_count"] = len(refusal_rows)

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_topic[row["topic"]].append(row)
    by_topic_summary: dict[str, dict] = {}
    for topic, rows in by_topic.items():
        scores = {"count": len(rows)}
        for metric in _METRICS:
            scores[metric] = _avg(rows, metric)
        by_topic_summary[topic] = scores

    return {"overall": overall, "by_topic": by_topic_summary}


def _print_summary(aggregates: dict) -> None:
    print("\n=== Eval results ===")
    overall = aggregates["overall"]
    print(f"Questions: {overall['count']}")
    for key, value in overall.items():
        if key == "count":
            continue
        print(f"  {key}: {value}")
    print("\nBy topic:")
    for topic, scores in aggregates["by_topic"].items():
        print(f"  {topic} (n={scores['count']})")
        for key, value in scores.items():
            if key == "count":
                continue
            print(f"    {key}: {value}")


def _load_dataset(limit: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in DATASET_PATH.read_text().splitlines()
        if line.strip()
    ]
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


async def _run(args: argparse.Namespace) -> int:
    _require_env("PINECONE_API_KEY")
    _require_env("PINECONE_INDEX_NAME")
    _require_env("OPENAI_API_KEY")
    _require_env("GROQ_API_KEY")

    rag_service = get_rag_service()
    if not rag_service.enabled:
        sys.stderr.write("\nRAG service is disabled — check PINECONE_* env vars.\n")
        return 2

    SessionLocal = _ephemeral_session_factory()
    db = SessionLocal()

    run_id = uuid.uuid4().hex[:12]
    user = User(
        email=f"evals+{run_id}@example.local",
        full_name=f"Eval Run {run_id}",
    )
    db.add(user)
    db.flush()
    thread = ChatThread(user_id=user.id, title="evals")
    db.add(thread)
    db.commit()
    db.refresh(user)
    db.refresh(thread)
    print(f"[setup] run_id={run_id} user_id={user.id} thread_id={thread.id}")

    chat = _build_groq_client("EVAL_ANSWER_MODEL")
    judge = _build_groq_client("EVAL_JUDGE_MODEL")

    print("[ingest] uploading fixtures...")
    attachments = _build_attachments()
    ingestion = await rag_service.ingest_attachments(
        db=db,
        user_id=user.id,
        thread_id=thread.id,
        message_id=None,
        attachments=attachments,
        preferred_provider_api_key_id=None,
    )
    if ingestion.notices:
        print("[ingest] notices:")
        for notice in ingestion.notices:
            print(f"  - {notice.filename}: {notice.message}")

    dataset = _load_dataset(args.limit)
    print(f"[run] {len(dataset)} questions")
    results: list[dict] = []
    try:
        for index, row in enumerate(dataset, 1):
            try:
                result = await _evaluate_question(
                    row=row,
                    db=db,
                    user=user,
                    thread=thread,
                    rag_service=rag_service,
                    chat=chat,
                    judge=judge,
                )
            except Exception as exc:
                print(f"  [{index}/{len(dataset)}] {row['id']} ERROR: {exc}")
                continue
            scores = result["scores"]
            print(
                f"  [{index}/{len(dataset)}] {row['id']:30s} "
                f"hit={scores['citation_hit_rate']:.2f} "
                f"prec={scores['context_precision']:.2f} "
                f"faith={scores['faithfulness']:.2f} "
                f"rel={scores['answer_relevance']:.2f}"
            )
            results.append(result)
    finally:
        print("[cleanup] deleting Pinecone vectors...")
        try:
            await rag_service.delete_thread_documents(
                db=db, user_id=user.id, thread_id=thread.id
            )
        except Exception:
            print("[cleanup] failed to delete Pinecone vectors (continuing)")
        db.close()

    if not results:
        print("\nNo results to report.")
        return 1

    aggregates = _aggregate(results)
    _print_summary(aggregates)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "answer_model": os.getenv("EVAL_ANSWER_MODEL", DEFAULT_MODEL),
        "judge_model": os.getenv("EVAL_JUDGE_MODEL", DEFAULT_MODEL),
        "limit": args.limit,
        "aggregates": aggregates,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[write] {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N dataset rows (useful for smoke tests).",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
