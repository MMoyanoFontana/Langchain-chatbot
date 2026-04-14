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
 * Splits a stream chunk on inline sentinels (CITATIONS and MESSAGE_ID).
 * Returns `{ text, citations, messageId }` where `text` is the displayable content.
 * Sentinels are always emitted at the very end of the stream, so we can strip
 * everything from the first sentinel onwards.
 */
export const splitInlineCitations = (
  chunk: string
): {
  text: string;
  citations: ConversationMessageCitation[];
  messageId: string | null;
  metrics: ConversationMessageMetrics | null;
} => {
  // Find whichever sentinel comes first.
  const candidates = [
    chunk.indexOf(CITATIONS_SENTINEL),
    chunk.indexOf(MESSAGE_ID_SENTINEL),
    chunk.indexOf(METRICS_SENTINEL),
  ].filter((idx) => idx !== -1);
  const firstIdx = candidates.length > 0 ? Math.min(...candidates) : -1;

  if (firstIdx === -1) {
    return { text: chunk, citations: [], messageId: null, metrics: null };
  }

  const text = chunk.slice(0, firstIdx);
  const tail = chunk.slice(firstIdx);

  let citations: ConversationMessageCitation[] = [];
  let messageId: string | null = null;
  let metrics: ConversationMessageMetrics | null = null;

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

  return { text, citations, messageId, metrics };
};
