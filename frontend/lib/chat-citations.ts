import type {
  ConversationMessageCitation,
  ConversationMessageMetrics,
} from "@/components/ai-elements/conversation";

export type BackendChatCitation = {
  document_id?: string | null;
  filename?: string | null;
  chunk_index?: number | null;
  score?: number | null;
  text?: string | null;
};

const normalizeOptionalText = (value: unknown): string | null => {
  if (typeof value !== "string") {
    return null;
  }

  const normalized = value.trim();
  return normalized || null;
};

const isDefined = <Value,>(value: Value | null): value is Value => value !== null;

export const toConversationCitation = (
  citation: BackendChatCitation | null | undefined
): ConversationMessageCitation | null => {
  if (!citation) {
    return null;
  }

  const documentId = normalizeOptionalText(citation.document_id);
  const text = normalizeOptionalText(citation.text);
  const chunkIndex =
    typeof citation.chunk_index === "number" &&
    Number.isInteger(citation.chunk_index) &&
    citation.chunk_index >= 0
      ? citation.chunk_index
      : null;
  const score =
    typeof citation.score === "number" && Number.isFinite(citation.score)
      ? citation.score
      : null;

  if (!documentId || chunkIndex === null || !text) {
    return null;
  }

  return {
    documentId,
    filename: normalizeOptionalText(citation.filename),
    chunkIndex,
    score,
    text,
  };
};

export const parseCitationHeader = (
  headerValue: string | null
): ConversationMessageCitation[] => {
  if (!headerValue) {
    return [];
  }

  try {
    const parsed = JSON.parse(headerValue);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed
      .map((citation) => toConversationCitation(citation as BackendChatCitation))
      .filter(isDefined);
  } catch {
    return [];
  }
};

export const CITATIONS_SENTINEL = "\x00CITATIONS:";
export const MESSAGE_ID_SENTINEL = "\x00MESSAGE_ID:";
export const METRICS_SENTINEL = "\x00METRICS:";
export const TOOL_CALL_SENTINEL = "\x00TOOL_CALL:";
export const TRACE_URL_SENTINEL = "\x00TRACE_URL:";
export const REASONING_SENTINEL = "\x00REASONING:";

type BackendMetricsPayload = {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  latency_ms?: number | null;
  time_to_first_token_ms?: number | null;
};

const toFiniteOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

export const toConversationMetrics = (
  payload: BackendMetricsPayload | null | undefined
): ConversationMessageMetrics | null => {
  if (!payload) return null;
  return {
    promptTokens: toFiniteOrNull(payload.prompt_tokens),
    completionTokens: toFiniteOrNull(payload.completion_tokens),
    totalTokens: toFiniteOrNull(payload.total_tokens),
    latencyMs: toFiniteOrNull(payload.latency_ms),
    timeToFirstTokenMs: toFiniteOrNull(payload.time_to_first_token_ms),
  };
};

/**
 * Splits a stream chunk on inline sentinels (CITATIONS, MESSAGE_ID, METRICS, TOOL_CALL).
 * Returns `{ text, citations, messageId, metrics, toolCalls }` where `text` is the
 * displayable content. CITATIONS/MESSAGE_ID/METRICS are always emitted at the end of
 * the stream. TOOL_CALL is emitted mid-stream before tool execution; it is stripped
 * out of the displayable text and returned separately.
 */
export const splitInlineCitations = (
  chunk: string
): {
  text: string;
  citations: ConversationMessageCitation[];
  messageId: string | null;
  metrics: ConversationMessageMetrics | null;
  toolCalls: Array<{ name: string; query: string }>;
  traceUrl: string | null;
  reasoningDeltas: string[];
} => {
  // Strip mid-stream TOOL_CALL events before applying end-of-stream sentinel logic.
  let strippedChunk = chunk;
  const toolCalls: Array<{ name: string; query: string }> = [];
  const reasoningDeltas: string[] = [];

  let tcIdx = strippedChunk.indexOf(TOOL_CALL_SENTINEL);
  while (tcIdx !== -1) {
    const before = strippedChunk.slice(0, tcIdx);
    const after = strippedChunk.slice(tcIdx + TOOL_CALL_SENTINEL.length);
    const nextSentinel = after.indexOf("\x00");
    const jsonPart = nextSentinel === -1 ? after : after.slice(0, nextSentinel);
    const remaining = nextSentinel === -1 ? "" : after.slice(nextSentinel);
    try {
      const parsed = JSON.parse(jsonPart);
      if (parsed?.name) {
        toolCalls.push({ name: String(parsed.name), query: parsed.query ? String(parsed.query) : "" });
      }
    } catch {
      // ignore parse error
    }
    strippedChunk = before + remaining;
    tcIdx = strippedChunk.indexOf(TOOL_CALL_SENTINEL);
  }

  // Strip mid-stream REASONING deltas. Payload is a JSON-encoded string so
  // the delta itself can safely contain any character (including \x00).
  let rIdx = strippedChunk.indexOf(REASONING_SENTINEL);
  while (rIdx !== -1) {
    const before = strippedChunk.slice(0, rIdx);
    const after = strippedChunk.slice(rIdx + REASONING_SENTINEL.length);
    const nextSentinel = after.indexOf("\x00");
    const jsonPart = nextSentinel === -1 ? after : after.slice(0, nextSentinel);
    const remaining = nextSentinel === -1 ? "" : after.slice(nextSentinel);
    try {
      const parsed = JSON.parse(jsonPart);
      if (typeof parsed === "string" && parsed.length > 0) {
        reasoningDeltas.push(parsed);
      }
    } catch {
      // ignore parse error
    }
    strippedChunk = before + remaining;
    rIdx = strippedChunk.indexOf(REASONING_SENTINEL);
  }

  // Find whichever end-of-stream sentinel comes first.
  const candidates = [
    strippedChunk.indexOf(CITATIONS_SENTINEL),
    strippedChunk.indexOf(MESSAGE_ID_SENTINEL),
    strippedChunk.indexOf(METRICS_SENTINEL),
    strippedChunk.indexOf(TRACE_URL_SENTINEL),
  ].filter((idx) => idx !== -1);
  const firstIdx = candidates.length > 0 ? Math.min(...candidates) : -1;

  if (firstIdx === -1) {
    return {
      text: strippedChunk,
      citations: [],
      messageId: null,
      metrics: null,
      toolCalls,
      traceUrl: null,
      reasoningDeltas,
    };
  }

  const text = strippedChunk.slice(0, firstIdx);
  const tail = strippedChunk.slice(firstIdx);

  let citations: ConversationMessageCitation[] = [];
  let messageId: string | null = null;
  let metrics: ConversationMessageMetrics | null = null;
  let traceUrl: string | null = null;

  const citSentinelIdx = tail.indexOf(CITATIONS_SENTINEL);
  if (citSentinelIdx !== -1) {
    const afterCit = tail.slice(citSentinelIdx + CITATIONS_SENTINEL.length);
    const nextSentinel = afterCit.indexOf("\x00");
    const jsonPart = nextSentinel === -1 ? afterCit : afterCit.slice(0, nextSentinel);
    try {
      const parsed = JSON.parse(jsonPart);
      citations = Array.isArray(parsed)
        ? parsed
            .map((c) => toConversationCitation(c as BackendChatCitation))
            .filter(isDefined)
        : [];
    } catch {
      // ignore parse error
    }
  }

  const msgSentinelIdx = tail.indexOf(MESSAGE_ID_SENTINEL);
  if (msgSentinelIdx !== -1) {
    const afterMsg = tail.slice(msgSentinelIdx + MESSAGE_ID_SENTINEL.length);
    const nextSentinel = afterMsg.indexOf("\x00");
    messageId = (nextSentinel === -1 ? afterMsg : afterMsg.slice(0, nextSentinel)).trim() || null;
  }

  const metricsSentinelIdx = tail.indexOf(METRICS_SENTINEL);
  if (metricsSentinelIdx !== -1) {
    const afterMetrics = tail.slice(metricsSentinelIdx + METRICS_SENTINEL.length);
    const nextSentinel = afterMetrics.indexOf("\x00");
    const jsonPart = nextSentinel === -1 ? afterMetrics : afterMetrics.slice(0, nextSentinel);
    try {
      const parsed = JSON.parse(jsonPart) as BackendMetricsPayload;
      metrics = toConversationMetrics(parsed);
    } catch {
      // ignore parse error
    }
  }

  const traceUrlSentinelIdx = tail.indexOf(TRACE_URL_SENTINEL);
  if (traceUrlSentinelIdx !== -1) {
    const afterTrace = tail.slice(traceUrlSentinelIdx + TRACE_URL_SENTINEL.length);
    const nextSentinel = afterTrace.indexOf("\x00");
    traceUrl = (nextSentinel === -1 ? afterTrace : afterTrace.slice(0, nextSentinel)).trim() || null;
  }

  return { text, citations, messageId, metrics, toolCalls, traceUrl, reasoningDeltas };
};
