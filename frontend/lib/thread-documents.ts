export type ThreadDocumentStatus = "pending" | "indexed" | "failed";

export type BackendThreadDocument = {
  id?: string | null;
  source_message_id?: string | null;
  filename?: string | null;
  media_type?: string | null;
  byte_size?: number | null;
  chunk_count?: number | null;
  status?: string | null;
  error_message?: string | null;
  indexed_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export interface ThreadDocument {
  id: string;
  sourceMessageId: string | null;
  filename: string | null;
  mediaType: string;
  byteSize: number;
  chunkCount: number;
  status: ThreadDocumentStatus;
  errorMessage: string | null;
  indexedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

const normalizeOptionalText = (value: unknown): string | null => {
  if (typeof value !== "string") {
    return null;
  }

  const normalized = value.trim();
  return normalized || null;
};

export const toThreadDocument = (
  document: BackendThreadDocument | null | undefined
): ThreadDocument | null => {
  if (!document) {
    return null;
  }

  const id = normalizeOptionalText(document.id);
  const mediaType = normalizeOptionalText(document.media_type);
  const status = normalizeOptionalText(document.status);
  const createdAt = normalizeOptionalText(document.created_at);
  const updatedAt = normalizeOptionalText(document.updated_at);
  const byteSize =
    typeof document.byte_size === "number" && Number.isFinite(document.byte_size)
      ? document.byte_size
      : null;
  const chunkCount =
    typeof document.chunk_count === "number" && Number.isFinite(document.chunk_count)
      ? document.chunk_count
      : null;

  if (
    !id ||
    !mediaType ||
    !createdAt ||
    !updatedAt ||
    byteSize === null ||
    chunkCount === null ||
    (status !== "pending" && status !== "indexed" && status !== "failed")
  ) {
    return null;
  }

  return {
    id,
    sourceMessageId: normalizeOptionalText(document.source_message_id),
    filename: normalizeOptionalText(document.filename),
    mediaType,
    byteSize,
    chunkCount,
    status,
    errorMessage: normalizeOptionalText(document.error_message),
    indexedAt: normalizeOptionalText(document.indexed_at),
    createdAt,
    updatedAt,
  };
};
