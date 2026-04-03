import type { ConversationMessageCitation } from "@/components/ai-elements/conversation";

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

/**
 * Splits a stream chunk on the inline citations sentinel.
 * Returns `{ text, citations }` where `text` is the displayable content
 * (everything before the sentinel) and `citations` is the parsed list
 * (empty if no sentinel was found).
 */
export const splitInlineCitations = (
  chunk: string
): { text: string; citations: ConversationMessageCitation[] } => {
  const idx = chunk.indexOf(CITATIONS_SENTINEL);
  if (idx === -1) {
    return { text: chunk, citations: [] };
  }

  const text = chunk.slice(0, idx);
  const jsonPart = chunk.slice(idx + CITATIONS_SENTINEL.length);
  try {
    const parsed = JSON.parse(jsonPart);
    const citations = Array.isArray(parsed)
      ? parsed
          .map((c) => toConversationCitation(c as BackendChatCitation))
          .filter(isDefined)
      : [];
    return { text, citations };
  } catch {
    return { text, citations: [] };
  }
};
