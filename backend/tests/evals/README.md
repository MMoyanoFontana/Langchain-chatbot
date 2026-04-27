# RAG eval harness

Custom eval suite that scores the system's retrieval and generation quality on a fixed dataset.

Each question is scored on five metrics:

| Metric | Type | What it measures |
|---|---|---|
| `citation_hit_rate` | deterministic | Of the source filenames we *expected* to be cited, how many showed up in the retrieved chunks |
| `context_precision` | LLM-judged | Of the retrieved chunks, how many are actually relevant to the question (per-chunk YES/NO, averaged) |
| `faithfulness` | LLM-judged | Are all claims in the answer supported by the retrieved context (0.0–1.0) |
| `answer_relevance` | LLM-judged | Does the answer address the question (0.0–1.0). A correct refusal counts as relevant. |
| `keyword_coverage` | deterministic | Fraction of expected substrings (key facts) that appear in the answer |

Plus a `refusal_accuracy` aggregate over the "should refuse / not in docs" subset.

## What's in the dataset

[`dataset.jsonl`](dataset.jsonl) — 18 questions across three topic groups:

- **`in_context`** (7) — single-chunk lookup: "What is the rate limit?", "What is a Quanto-mesh?"
- **`synthesis`** (6) — needs multiple chunks or cross-document reasoning: "Walk me through the full handshake", "How does auth work and what to do when the token expires?"
- **`refusal`** (5) — answer is *not* in the data and the model should say so: "What is the rate limit for the OpenAI API?", "Who founded Orbital Widgets?"

Data lives in [`fixtures/`](fixtures/) —consisting of three fictional documents (an API reference, a protocol spec, and a glossary). These were created specifically for this evaluation to ensure the judge’s decisions rely solely on the provided context eliminating the influence of prior knowledge from its training phase.


## Setup

The harness reuses the backend's `.env` for `PINECONE_*` and `OPENAI_API_KEY`, and adds:

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | yes | Free tier from console.groq.com — used for both answer generation and judging |
| `EVAL_ANSWER_MODEL` | no | Defaults to `llama-3.3-70b-versatile` |
| `EVAL_JUDGE_MODEL` | no | Defaults to `llama-3.3-70b-versatile` |
| `PINECONE_NAMESPACE_PREFIX_EVAL` | no | Defaults to `evals` — keeps eval data isolated from prod `rag` namespaces |

The runner forces `PINECONE_NAMESPACE_PREFIX=evals` (overriding the .env) so an eval run never writes to the real namespace.

## Run

```bash
cd backend
uv run python -m tests.evals.run_evals              # full run (18 questions)
uv run python -m tests.evals.run_evals --limit 3    # smoke run
```

Each full run does:

1. Ephemeral SQLite DB + fresh `User` + `ChatThread` rows
2. Ingest every fixture into Pinecone (real embeddings via OpenAI)
3. For each question: retrieve → generate answer → score with the judge
4. Print a per-question line + an aggregate summary
5. Write a JSON report to `results/{UTC-timestamp}.json`
6. Delete the Pinecone vectors that were just ingested

## Output

Per-run JSON in [`results/`](results/). Format:

```json
{
  "run_id": "abc123",
  "timestamp": "2026-04-27T...",
  "answer_model": "llama-3.3-70b-versatile",
  "judge_model": "llama-3.3-70b-versatile",
  "aggregates": {
    "overall": {"count": 18, "citation_hit_rate": 0.94, "faithfulness": 0.91, ...},
    "by_topic": {
      "in_context": {...},
      "synthesis": {...},
      "refusal": {...}
    }
  },
  "results": [
    {"id": "ow-rate-limit", "question": "...", "answer": "...", "scores": {...}, ...}
  ]
}
```

The console summary prints the same aggregates so you don't have to open the file.


## Adding a question

Append a JSON object to `dataset.jsonl` with these fields:

```json
{
  "id": "unique-slug",
  "topic": "in_context | synthesis | refusal",
  "question": "...",
  "expected_citation_filenames": ["orbital_widgets_api.md"],
  "expected_keywords": ["120", "minute"],
  "must_refuse": false
}
```

For `refusal` questions: leave `expected_citation_filenames` and `expected_keywords` empty and set `must_refuse: true`.

## Extending

- **New metric** → add a function to [`scorers.py`](scorers.py), call it from `_evaluate_question` in [`run_evals.py`](run_evals.py), add the key to `_METRICS`.
- **New corpus** → drop a `.md` file into [`fixtures/`](fixtures/) and add questions that cite it.
- **Different judge** → set `EVAL_JUDGE_MODEL` (Groq) or fork `_build_groq_client` to point at OpenAI/Anthropic.
