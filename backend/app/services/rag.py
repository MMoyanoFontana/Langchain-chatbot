from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, TypedDict
from urllib.parse import unquote_to_bytes

from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    ChatMessage,
    ChatThread,
    DocumentIndexStatus,
    IndexedDocument,
    Provider,
    ProviderApiKey,
    ProviderCode,
    utc_now,
)
from app.security import EncryptionConfigError, decrypt_secret
from app.services.pinecone_store import (
    PineconeVectorRecord,
    PineconeVectorStore,
    PineconeVectorStoreError,
)

LOGGER = logging.getLogger(__name__)

DATA_URL_PATTERN = re.compile(
    r"^data:(?P<media_type>[^;,]+)?(?P<base64>;base64)?,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)
TEXTUAL_MEDIA_TYPE_PREFIXES = ("text/",)
SUPPORTED_TEXTUAL_MEDIA_TYPES = {
    "application/csv",
    "application/json",
    "application/ld+json",
    "application/pdf",
    "application/xml",
}
SMALL_TALK_PROMPTS = {
    "hello",
    "hi",
    "hey",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "cool",
    "great",
    "sounds good",
}


class RagAttachment(TypedDict):
    type: str
    filename: str | None
    media_type: str
    url: str


class EmbeddingsClient(Protocol):
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    async def aembed_query(self, text: str) -> list[float]:
        ...


class RagConfigurationError(RuntimeError):
    pass


class AttachmentDecodeError(RuntimeError):
    pass


class AttachmentParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class RagSettings:
    enabled: bool
    pinecone_api_key: str | None
    pinecone_index_name: str | None
    namespace_prefix: str
    embedding_model: str
    embedding_dimensions: int | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chars: int
    min_relevance_score: float
    max_file_bytes: int


@dataclass(frozen=True)
class DecodedAttachment:
    filename: str | None
    media_type: str
    raw_bytes: bytes
    checksum_sha256: str
    byte_size: int


@dataclass(frozen=True)
class IngestionNotice:
    filename: str | None
    message: str


@dataclass(frozen=True)
class IngestionResult:
    notices: tuple[IngestionNotice, ...] = ()


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    filename: str | None
    chunk_index: int
    score: float | None
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    thread_has_documents: bool
    chunks: tuple[RetrievedChunk, ...] = ()
    notice: str | None = None


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_text_content(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _safe_notice(filename: str | None, message: str) -> IngestionNotice:
    return IngestionNotice(filename=_normalize_optional_text(filename), message=message)


def load_rag_settings() -> RagSettings:
    pinecone_api_key = _normalize_optional_text(os.getenv("PINECONE_API_KEY"))
    pinecone_index_name = _normalize_optional_text(os.getenv("PINECONE_INDEX_NAME"))
    namespace_prefix = _normalize_optional_text(os.getenv("PINECONE_NAMESPACE_PREFIX")) or "rag"
    chunk_size = max(int(os.getenv("RAG_CHUNK_SIZE", "1400")), 400)
    chunk_overlap = max(int(os.getenv("RAG_CHUNK_OVERLAP", "200")), 0)
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    return RagSettings(
        enabled=bool(pinecone_api_key and pinecone_index_name),
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        namespace_prefix=namespace_prefix,
        embedding_model=_normalize_optional_text(os.getenv("RAG_EMBEDDING_MODEL")) or "text-embedding-3-small",
        embedding_dimensions=(
            int(raw_dimensions)
            if (raw_dimensions := _normalize_optional_text(os.getenv("RAG_EMBEDDING_DIMENSIONS"))) is not None
            else None
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=max(int(os.getenv("RAG_TOP_K", "4")), 1),
        max_context_chars=max(int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000")), 1000),
        min_relevance_score=float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.2")),
        max_file_bytes=max(int(os.getenv("RAG_MAX_FILE_BYTES", str(5 * 1024 * 1024))), 1024),
    )


def _decode_data_url(attachment: RagAttachment, *, max_file_bytes: int) -> DecodedAttachment:
    raw_url = attachment["url"]
    if not isinstance(raw_url, str):
        raise AttachmentDecodeError("Attachment URL is missing.")

    match = DATA_URL_PATTERN.match(raw_url.strip())
    if match is None:
        raise AttachmentDecodeError("Attachment data URL is invalid or unsupported.")

    declared_media_type = _normalize_optional_text(match.group("media_type"))
    media_type = _normalize_optional_text(attachment.get("media_type")) or declared_media_type
    if media_type is None:
        raise AttachmentDecodeError("Attachment media type is missing.")

    encoded_payload = match.group("data")
    try:
        if match.group("base64"):
            raw_bytes = base64.b64decode(encoded_payload, validate=True)
        else:
            raw_bytes = unquote_to_bytes(encoded_payload)
    except (ValueError, binascii.Error) as exc:
        raise AttachmentDecodeError("Attachment payload could not be decoded.") from exc

    if not raw_bytes:
        raise AttachmentDecodeError("Attachment payload is empty.")
    if len(raw_bytes) > max_file_bytes:
        raise AttachmentDecodeError(
            f"Attachment exceeds the {max_file_bytes // (1024 * 1024)} MB indexing limit."
        )

    return DecodedAttachment(
        filename=_normalize_optional_text(attachment.get("filename")),
        media_type=media_type,
        raw_bytes=raw_bytes,
        checksum_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        byte_size=len(raw_bytes),
    )


def _decode_text_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AttachmentParseError("Attachment text could not be decoded.")


def _extract_text(decoded: DecodedAttachment) -> str:
    media_type = decoded.media_type.lower()
    if media_type == "application/pdf":
        reader = PdfReader(io.BytesIO(decoded.raw_bytes))
        parts: list[str] = []
        for index, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            parts.append(f"[Page {index + 1}]\n{page_text}")
        text = "\n\n".join(parts)
        if not text.strip():
            raise AttachmentParseError("PDF text extraction returned no text.")
        return _normalize_text_content(text)

    if media_type in {"application/json", "application/ld+json"}:
        try:
            parsed = json.loads(_decode_text_bytes(decoded.raw_bytes))
        except json.JSONDecodeError as exc:
            raise AttachmentParseError("JSON parsing failed.") from exc
        return _normalize_text_content(json.dumps(parsed, ensure_ascii=False, indent=2))

    if media_type.startswith(TEXTUAL_MEDIA_TYPE_PREFIXES) or media_type in SUPPORTED_TEXTUAL_MEDIA_TYPES:
        text = _decode_text_bytes(decoded.raw_bytes)
        normalized = _normalize_text_content(text)
        if not normalized:
            raise AttachmentParseError("Attachment does not contain readable text.")
        return normalized

    raise AttachmentParseError(f"Unsupported attachment media type `{decoded.media_type}`.")


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = _normalize_text_content(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end
        if end < text_length:
            split_at = normalized.rfind("\n\n", start, target_end)
            if split_at > start:
                end = split_at
            else:
                split_at = normalized.rfind(". ", start, target_end)
                if split_at > start:
                    end = split_at + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def _build_namespace(settings: RagSettings, user_id: str) -> str:
    compact_user_id = re.sub(r"[^a-zA-Z0-9_-]", "-", user_id)
    return f"{settings.namespace_prefix}-{compact_user_id}"


def _thread_has_small_talk_prompt(prompt: str) -> bool:
    normalized = prompt.strip().lower()
    return normalized in SMALL_TALK_PROMPTS or len(normalized) <= 2


def _build_retrieval_filter(thread_id: str) -> dict[str, Any]:
    return {"thread_id": {"$eq": thread_id}}


def _sort_and_trim_chunks(
    chunks: list[RetrievedChunk],
    *,
    max_context_chars: int,
) -> tuple[RetrievedChunk, ...]:
    ordered = sorted(chunks, key=lambda item: item.score or 0.0, reverse=True)
    selected: list[RetrievedChunk] = []
    total_chars = 0

    for chunk in ordered:
        text = chunk.text.strip()
        if not text:
            continue

        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining].rstrip()
        if not text:
            continue

        selected.append(
            RetrievedChunk(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                text=text,
            )
        )
        total_chars += len(text)

    return tuple(selected)


def _classify_indexing_failure(exc: Exception) -> tuple[str, str]:
    normalized = str(exc).lower()
    if "dimension" in normalized and "does not match" in normalized:
        return (
            "Pinecone index dimension does not match the embedding size.",
            "Document indexing is temporarily misconfigured on the server.",
        )
    if "api key" in normalized or "unauthorized" in normalized:
        return (
            "Embedding or Pinecone credentials were rejected.",
            "Document indexing could not authenticate with the configured backend services.",
        )
    return (
        "Document indexing failed.",
        "Document indexing failed. You can still chat, but retrieval is unavailable for this file.",
    )


class RagService:
    def __init__(
        self,
        *,
        settings: RagSettings,
        vector_store: PineconeVectorStore | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store

    @property
    def enabled(self) -> bool:
        return self._settings.enabled and self._vector_store is not None

    async def ingest_attachments(
        self,
        *,
        db: Session,
        user_id: str,
        thread_id: str,
        message_id: str | None,
        attachments: list[RagAttachment],
        preferred_provider_api_key_id: str | None,
    ) -> IngestionResult:
        if not self.enabled or not attachments:
            return IngestionResult()

        notices: list[IngestionNotice] = []
        pending_documents: list[tuple[IndexedDocument, DecodedAttachment, list[str]]] = []

        for attachment in attachments:
            try:
                decoded = _decode_data_url(attachment, max_file_bytes=self._settings.max_file_bytes)
            except AttachmentDecodeError as exc:
                notices.append(_safe_notice(attachment.get("filename"), str(exc)))
                LOGGER.warning(
                    "rag_attachment_decode_failed thread_id=%s user_id=%s filename=%s error=%s",
                    thread_id,
                    user_id,
                    _normalize_optional_text(attachment.get("filename")) or "<unnamed>",
                    exc,
                )
                continue

            existing_document = db.scalar(
                select(IndexedDocument).where(
                    IndexedDocument.user_id == user_id,
                    IndexedDocument.thread_id == thread_id,
                    IndexedDocument.checksum_sha256 == decoded.checksum_sha256,
                )
            )
            if existing_document is not None and existing_document.status == DocumentIndexStatus.INDEXED:
                if existing_document.source_message_id is None and message_id is not None:
                    existing_document.source_message_id = message_id
                    db.commit()
                LOGGER.info(
                    "rag_attachment_reused thread_id=%s user_id=%s document_id=%s filename=%s",
                    thread_id,
                    user_id,
                    existing_document.id,
                    existing_document.filename or "<unnamed>",
                )
                continue

            document = existing_document or IndexedDocument(
                user_id=user_id,
                thread_id=thread_id,
                source_message_id=message_id,
                filename=decoded.filename,
                media_type=decoded.media_type,
                checksum_sha256=decoded.checksum_sha256,
                byte_size=decoded.byte_size,
                chunk_count=0,
                pinecone_namespace=_build_namespace(self._settings, user_id),
                status=DocumentIndexStatus.PENDING,
                error_message=None,
                indexed_at=None,
            )
            if existing_document is None:
                db.add(document)

            document.source_message_id = document.source_message_id or message_id
            document.filename = decoded.filename or document.filename
            document.media_type = decoded.media_type
            document.byte_size = decoded.byte_size
            document.pinecone_namespace = _build_namespace(self._settings, user_id)
            document.status = DocumentIndexStatus.PENDING
            document.error_message = None
            db.commit()
            db.refresh(document)

            try:
                text = _extract_text(decoded)
                chunks = _chunk_text(
                    text,
                    chunk_size=self._settings.chunk_size,
                    chunk_overlap=self._settings.chunk_overlap,
                )
                if not chunks:
                    raise AttachmentParseError("Attachment produced no indexable text chunks.")
            except AttachmentParseError as exc:
                document.status = DocumentIndexStatus.FAILED
                document.error_message = str(exc)
                document.chunk_count = 0
                document.indexed_at = None
                db.commit()
                notices.append(_safe_notice(decoded.filename, str(exc)))
                LOGGER.warning(
                    "rag_attachment_parse_failed thread_id=%s user_id=%s document_id=%s filename=%s error=%s",
                    thread_id,
                    user_id,
                    document.id,
                    decoded.filename or "<unnamed>",
                    exc,
                )
                continue

            pending_documents.append((document, decoded, chunks))

        if not pending_documents:
            return IngestionResult(tuple(notices))

        try:
            embeddings_client = await self._build_embeddings_client(
                db=db,
                user_id=user_id,
                preferred_provider_api_key_id=preferred_provider_api_key_id,
            )
        except RagConfigurationError as exc:
            for document, decoded, _chunks in pending_documents:
                document.status = DocumentIndexStatus.FAILED
                document.error_message = str(exc)
                document.chunk_count = 0
                document.indexed_at = None
                db.commit()
                notices.append(_safe_notice(decoded.filename, str(exc)))
            LOGGER.warning(
                "rag_embedding_config_missing thread_id=%s user_id=%s error=%s",
                thread_id,
                user_id,
                exc,
            )
            return IngestionResult(tuple(notices))

        for document, decoded, chunks in pending_documents:
            try:
                vectors = await self._build_vectors(
                    document=document,
                    decoded=decoded,
                    chunks=chunks,
                    embeddings_client=embeddings_client,
                )
                await self._vector_store.upsert(
                    namespace=document.pinecone_namespace,
                    vectors=vectors,
                )
            except (PineconeVectorStoreError, Exception) as exc:
                error_message, notice_message = _classify_indexing_failure(exc)
                document.status = DocumentIndexStatus.FAILED
                document.error_message = error_message
                document.chunk_count = 0
                document.indexed_at = None
                db.commit()
                notices.append(_safe_notice(decoded.filename, notice_message))
                LOGGER.exception(
                    "rag_document_index_failed thread_id=%s user_id=%s document_id=%s filename=%s",
                    thread_id,
                    user_id,
                    document.id,
                    decoded.filename or "<unnamed>",
                )
                continue

            document.status = DocumentIndexStatus.INDEXED
            document.error_message = None
            document.chunk_count = len(chunks)
            document.indexed_at = utc_now()
            db.commit()
            LOGGER.info(
                "rag_document_indexed thread_id=%s user_id=%s document_id=%s filename=%s chunks=%s",
                thread_id,
                user_id,
                document.id,
                decoded.filename or "<unnamed>",
                len(chunks),
            )

        return IngestionResult(tuple(notices))

    async def retrieve(
        self,
        *,
        db: Session,
        user_id: str,
        thread_id: str,
        prompt: str,
        preferred_provider_api_key_id: str | None,
    ) -> RetrievalResult:
        indexed_document_count = db.scalar(
            select(func.count())
            .select_from(IndexedDocument)
            .where(
                IndexedDocument.user_id == user_id,
                IndexedDocument.thread_id == thread_id,
                IndexedDocument.status == DocumentIndexStatus.INDEXED,
            )
        )
        thread_has_documents = bool(indexed_document_count)
        if not self.enabled or not thread_has_documents:
            return RetrievalResult(thread_has_documents=thread_has_documents)

        normalized_prompt = prompt.strip()
        if not normalized_prompt or _thread_has_small_talk_prompt(normalized_prompt):
            return RetrievalResult(thread_has_documents=True)

        try:
            embeddings_client = await self._build_embeddings_client(
                db=db,
                user_id=user_id,
                preferred_provider_api_key_id=preferred_provider_api_key_id,
            )
            query_vector = await embeddings_client.aembed_query(normalized_prompt)
            matches = await self._vector_store.query(
                namespace=_build_namespace(self._settings, user_id),
                query_vector=query_vector,
                top_k=self._settings.top_k,
                metadata_filter=_build_retrieval_filter(thread_id),
            )
        except RagConfigurationError as exc:
            LOGGER.warning(
                "rag_query_skipped_missing_embeddings thread_id=%s user_id=%s error=%s",
                thread_id,
                user_id,
                exc,
            )
            return RetrievalResult(
                thread_has_documents=True,
                notice="Uploaded files exist in this chat, but retrieval is unavailable right now.",
            )
        except PineconeVectorStoreError:
            LOGGER.exception(
                "rag_query_failed thread_id=%s user_id=%s",
                thread_id,
                user_id,
            )
            return RetrievalResult(
                thread_has_documents=True,
                notice="Uploaded files are indexed for this chat, but retrieval failed for this question.",
            )

        relevant_chunks: list[RetrievedChunk] = []
        for match in matches:
            if match.score is not None and match.score < self._settings.min_relevance_score:
                continue

            text = match.metadata.get("text")
            document_id = match.metadata.get("document_id")
            chunk_index = match.metadata.get("chunk_index")
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(document_id, str) or not document_id:
                continue
            if isinstance(chunk_index, float) and chunk_index.is_integer():
                chunk_index = int(chunk_index)
            if not isinstance(chunk_index, int):
                continue

            filename = match.metadata.get("filename")
            relevant_chunks.append(
                RetrievedChunk(
                    document_id=document_id,
                    filename=filename if isinstance(filename, str) else None,
                    chunk_index=chunk_index,
                    score=match.score,
                    text=_normalize_text_content(text),
                )
            )

        trimmed_chunks = _sort_and_trim_chunks(
            relevant_chunks,
            max_context_chars=self._settings.max_context_chars,
        )
        if not trimmed_chunks:
            return RetrievalResult(
                thread_has_documents=True,
                notice="No relevant uploaded-document excerpts were retrieved for this question.",
            )

        return RetrievalResult(thread_has_documents=True, chunks=trimmed_chunks)

    async def delete_thread_documents(
        self,
        *,
        db: Session,
        user_id: str,
        thread_id: str,
    ) -> None:
        if not self.enabled:
            return
        documents = list(
            db.scalars(
                select(IndexedDocument).where(
                    IndexedDocument.user_id == user_id,
                    IndexedDocument.thread_id == thread_id,
                )
            ).all()
        )
        if not documents:
            return

        namespace = documents[0].pinecone_namespace or _build_namespace(self._settings, user_id)
        try:
            await self._vector_store.delete_by_filter(
                namespace=namespace,
                metadata_filter=_build_retrieval_filter(thread_id),
            )
        except PineconeVectorStoreError:
            LOGGER.exception(
                "rag_thread_cleanup_failed thread_id=%s user_id=%s",
                thread_id,
                user_id,
            )

    async def delete_user_documents(
        self,
        *,
        db: Session,
        user_id: str,
    ) -> None:
        if not self.enabled:
            return
        has_documents = db.scalar(
            select(func.count())
            .select_from(IndexedDocument)
            .where(IndexedDocument.user_id == user_id)
        )
        if not has_documents:
            return

        try:
            await self._vector_store.delete_by_filter(
                namespace=_build_namespace(self._settings, user_id),
                metadata_filter={"user_id": {"$eq": user_id}},
            )
        except PineconeVectorStoreError:
            LOGGER.exception("rag_user_cleanup_failed user_id=%s", user_id)

    async def delete_document(
        self,
        *,
        db: Session,
        user_id: str,
        thread_id: str,
        document_id: str,
    ) -> None:
        document = db.scalar(
            select(IndexedDocument).where(
                IndexedDocument.id == document_id,
                IndexedDocument.user_id == user_id,
                IndexedDocument.thread_id == thread_id,
            )
        )
        if document is None:
            return

        if self.enabled and document.status == DocumentIndexStatus.INDEXED:
            try:
                await self._vector_store.delete_by_filter(
                    namespace=document.pinecone_namespace or _build_namespace(self._settings, user_id),
                    metadata_filter={"document_id": {"$eq": document_id}},
                )
            except PineconeVectorStoreError:
                LOGGER.exception(
                    "rag_document_delete_failed document_id=%s user_id=%s",
                    document_id,
                    user_id,
                )
                document.status = DocumentIndexStatus.FAILED
                document.error_message = "Vector deletion failed; pending retry."
                db.commit()
                return

        db.delete(document)
        db.commit()
        LOGGER.info(
            "rag_document_deleted document_id=%s thread_id=%s user_id=%s",
            document_id,
            thread_id,
            user_id,
        )

    async def retry_document(
        self,
        *,
        db: Session,
        user_id: str,
        thread_id: str,
        document_id: str,
    ) -> IndexedDocument:
        document = db.scalar(
            select(IndexedDocument).where(
                IndexedDocument.id == document_id,
                IndexedDocument.user_id == user_id,
                IndexedDocument.thread_id == thread_id,
            )
        )
        if document is None:
            raise RagConfigurationError("Document not found.")

        if document.source_message_id is None:
            raise RagConfigurationError(
                "Original file data is unavailable — document has no source message."
            )

        message = db.scalar(
            select(ChatMessage).where(ChatMessage.id == document.source_message_id)
        )
        if message is None:
            raise RagConfigurationError(
                "Original file data is unavailable — source message was deleted."
            )

        matching_attachment: RagAttachment | None = None
        for raw in message.attachments or []:
            url = (raw.get("url") or "").strip()
            media_type = (raw.get("media_type") or "").strip()
            if not url or not media_type:
                continue
            candidate: RagAttachment = {
                "type": "file",
                "filename": raw.get("filename"),
                "media_type": media_type,
                "url": url,
            }
            try:
                decoded = _decode_data_url(candidate, max_file_bytes=self._settings.max_file_bytes)
            except AttachmentDecodeError:
                continue
            if decoded.checksum_sha256 == document.checksum_sha256:
                matching_attachment = candidate
                break

        if matching_attachment is None:
            raise RagConfigurationError(
                "Original file data is unavailable — no matching attachment found in source message."
            )

        thread = db.get(ChatThread, thread_id)
        preferred_key_id = thread.provider_api_key_id if thread else None

        await self.ingest_attachments(
            db=db,
            user_id=user_id,
            thread_id=thread_id,
            message_id=document.source_message_id,
            attachments=[matching_attachment],
            preferred_provider_api_key_id=preferred_key_id,
        )

        db.refresh(document)
        return document

    def build_augmented_prompt(
        self,
        *,
        base_prompt: str,
        ingestion_result: IngestionResult,
        retrieval_result: RetrievalResult,
    ) -> str:
        parts = [base_prompt.strip()]

        if ingestion_result.notices:
            lines = ["Document indexing notices:"]
            for notice in ingestion_result.notices:
                label = notice.filename or "Attachment"
                lines.append(f"- {label}: {notice.message}")
            parts.append("\n".join(lines))

        if retrieval_result.chunks:
            lines = [
                "Retrieved uploaded-document context:",
                "Use these excerpts as the primary source of truth for claims about uploaded files.",
            ]
            for index, chunk in enumerate(retrieval_result.chunks, start=1):
                label = chunk.filename or f"Document {chunk.document_id}"
                lines.append(f"[{index}] {label} (chunk {chunk.chunk_index})")
                lines.append(chunk.text)
            parts.append("\n".join(lines))
        elif retrieval_result.notice:
            parts.append(
                "\n".join(
                    [
                        "Uploaded-document grounding note:",
                        retrieval_result.notice,
                        "If the answer depends on uploaded files, say so plainly instead of guessing.",
                    ]
                )
            )

        return "\n\n".join(part for part in parts if part)

    async def _build_vectors(
        self,
        *,
        document: IndexedDocument,
        decoded: DecodedAttachment,
        chunks: list[str],
        embeddings_client: EmbeddingsClient,
    ) -> list[PineconeVectorRecord]:
        embeddings = await embeddings_client.aembed_documents(chunks)
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding response length did not match chunk count.")

        uploaded_at = (document.created_at or utc_now()).isoformat()
        vectors: list[PineconeVectorRecord] = []
        for chunk_index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            vectors.append(
                PineconeVectorRecord(
                    id=f"{document.id}:{chunk_index}",
                    values=[float(value) for value in embedding],
                    metadata={
                        "document_id": document.id,
                        "user_id": document.user_id,
                        "thread_id": document.thread_id,
                        "filename": decoded.filename or "",
                        "media_type": decoded.media_type,
                        "checksum_sha256": decoded.checksum_sha256,
                        "chunk_index": chunk_index,
                        "uploaded_at": uploaded_at,
                        "text": chunk_text,
                    },
                )
            )
        return vectors

    async def _build_embeddings_client(
        self,
        *,
        db: Session,
        user_id: str,
        preferred_provider_api_key_id: str | None,
    ) -> EmbeddingsClient:
        api_key = _normalize_optional_text(os.getenv("OPENAI_API_KEY"))
        if api_key is None:
            api_key = self._resolve_user_openai_api_key(
                db=db,
                user_id=user_id,
                preferred_provider_api_key_id=preferred_provider_api_key_id,
            )
        if api_key is None:
            raise RagConfigurationError(
                "No OpenAI embedding key is available. Configure `OPENAI_API_KEY` or add an active OpenAI provider key."
            )

        dimensions = self._settings.embedding_dimensions
        if dimensions is None and self._vector_store is not None:
            try:
                dimensions = await self._vector_store.describe_index_dimension()
            except PineconeVectorStoreError as exc:
                raise RagConfigurationError("Pinecone index dimension could not be resolved.") from exc

        return OpenAIEmbeddings(
            model=self._settings.embedding_model,
            api_key=api_key,
            dimensions=dimensions,
        )

    def _resolve_user_openai_api_key(
        self,
        *,
        db: Session,
        user_id: str,
        preferred_provider_api_key_id: str | None,
    ) -> str | None:
        preferred_key: ProviderApiKey | None = None
        if preferred_provider_api_key_id:
            preferred_key = db.scalar(
                select(ProviderApiKey)
                .join(Provider, Provider.id == ProviderApiKey.provider_id)
                .where(
                    ProviderApiKey.id == preferred_provider_api_key_id,
                    ProviderApiKey.user_id == user_id,
                    ProviderApiKey.is_active.is_(True),
                    Provider.code == ProviderCode.OPENAI,
                    Provider.is_active.is_(True),
                )
            )

        openai_key = preferred_key or db.scalar(
            select(ProviderApiKey)
            .join(Provider, Provider.id == ProviderApiKey.provider_id)
            .where(
                ProviderApiKey.user_id == user_id,
                ProviderApiKey.is_active.is_(True),
                Provider.code == ProviderCode.OPENAI,
                Provider.is_active.is_(True),
            )
            .order_by(ProviderApiKey.is_default.desc(), ProviderApiKey.updated_at.desc())
        )
        if openai_key is None:
            return None

        try:
            return decrypt_secret(openai_key.encrypted_api_key)
        except (EncryptionConfigError, ValueError) as exc:
            raise RagConfigurationError("Stored OpenAI provider key could not be decrypted.") from exc


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    settings = load_rag_settings()
    if not settings.enabled or settings.pinecone_api_key is None or settings.pinecone_index_name is None:
        return RagService(settings=settings, vector_store=None)
    return RagService(
        settings=settings,
        vector_store=PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
        ),
    )
