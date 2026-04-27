"""Scoring functions for the RAG eval harness.

Four metrics, all self-made:

- ``citation_hit_rate`` — fraction of expected source filenames that appear in retrieval output.
- ``context_precision`` — LLM-judged: of retrieved chunks, how many are relevant to the question.
- ``faithfulness`` — LLM-judged: are all claims in the answer supported by retrieved context.
- ``answer_relevance`` — LLM-judged: does the answer address the question.

Plus two cheap deterministic helpers used alongside the LLM scores:

- ``keyword_coverage`` — fraction of expected substrings that appear in the answer.
- ``is_refusal`` — substring check used to score "should refuse / not in docs" cases.
"""

from __future__ import annotations

import re

from app.services.rag import RetrievedChunk


_REFUSAL_HINTS: tuple[str, ...] = (
    "i don't",
    "i do not",
    "i'm not able",
    "no information",
    "not in the",
    "cannot find",
    "not provided",
    "unable to find",
    "isn't covered",
    "is not covered",
    "is not included",
    "not mentioned",
    "no mention",
    "the context does not",
    "context doesn't",
    "documents do not",
    "documents don't",
)


def is_refusal(answer: str) -> bool:
    """True if the answer reads as a refusal to answer / acknowledgement of insufficient context."""
    lowered = answer.lower()
    return any(hint in lowered for hint in _REFUSAL_HINTS)


def citation_hit_rate(
    retrieved: list[RetrievedChunk], expected_filenames: list[str]
) -> float:
    """Fraction of expected filenames that appear among retrieved chunk filenames."""
    if not expected_filenames:
        return 1.0
    retrieved_filenames = {chunk.filename for chunk in retrieved if chunk.filename}
    hits = sum(1 for filename in expected_filenames if filename in retrieved_filenames)
    return hits / len(expected_filenames)


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected substrings (case-insensitive) that appear in the answer."""
    if not expected_keywords:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return hits / len(expected_keywords)


_SCORE_PATTERN = re.compile(r"\d(?:\.\d+)?")


def _parse_score(text: str) -> float:
    match = _SCORE_PATTERN.search(text)
    if match is None:
        return 0.0
    try:
        score = float(match.group(0))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, score))


CONTEXT_PRECISION_PROMPT = """You are evaluating whether a retrieved passage is relevant to a question.

QUESTION: {question}

PASSAGE:
{passage}

Is this passage relevant to answering the question? Reply with exactly one word: YES or NO."""


FAITHFULNESS_PROMPT = """Determine whether every factual claim in the ANSWER is directly supported by the CONTEXT.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Score the answer's faithfulness on a scale from 0.0 to 1.0:
- 1.0 = every claim is supported by the context, OR the answer correctly states the context does not cover the question
- 0.5 = some claims are supported, others are not in the context
- 0.0 = answer contradicts or hallucinates beyond the context

Reply with only the score (a single number)."""


ANSWER_RELEVANCE_PROMPT = """Determine whether the ANSWER addresses the QUESTION.

QUESTION: {question}

ANSWER: {answer}

Score the answer's relevance on a scale from 0.0 to 1.0:
- 1.0 = directly addresses the question (a refusal that explains the question cannot be answered also scores 1.0)
- 0.5 = partially addresses the question
- 0.0 = off-topic or non-responsive

Reply with only the score (a single number)."""


async def context_precision(
    *, question: str, retrieved: list[RetrievedChunk], judge
) -> float:
    if not retrieved:
        return 0.0
    scores: list[float] = []
    for chunk in retrieved:
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, passage=chunk.text)
        response = await judge.ainvoke(prompt)
        verdict = (response.content or "").strip().upper()
        scores.append(1.0 if verdict.startswith("YES") else 0.0)
    return sum(scores) / len(scores)


async def faithfulness(
    *, question: str, answer: str, retrieved: list[RetrievedChunk], judge
) -> tuple[float, str]:
    context = (
        "\n\n---\n\n".join(chunk.text for chunk in retrieved)
        if retrieved
        else "(no context retrieved)"
    )
    prompt = FAITHFULNESS_PROMPT.format(
        context=context, question=question, answer=answer
    )
    response = await judge.ainvoke(prompt)
    raw = (response.content or "").strip()
    return _parse_score(raw), raw


async def answer_relevance(
    *, question: str, answer: str, judge
) -> tuple[float, str]:
    prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    response = await judge.ainvoke(prompt)
    raw = (response.content or "").strip()
    return _parse_score(raw), raw
