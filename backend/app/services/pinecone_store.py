from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone
from pinecone.exceptions import PineconeException

PineconeMetadataValue = str | int | float | bool


class PineconeVectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class PineconeVectorRecord:
    id: str
    values: list[float]
    metadata: dict[str, PineconeMetadataValue]


@dataclass(frozen=True)
class PineconeQueryMatch:
    id: str
    score: float | None
    metadata: dict[str, Any]


def _normalize_metadata(raw_metadata: object) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    return {}


class PineconeVectorStore:
    def __init__(self, *, api_key: str, index_name: str) -> None:
        self._client = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._index = None
        self._dimension: int | None = None

    def _get_index(self):
        if self._index is None:
            self._index = self._client.Index(self._index_name)
        return self._index

    async def describe_index_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension

        description = await asyncio.to_thread(
            self._client.describe_index,
            self._index_name,
        )
        dimension = getattr(description, "dimension", None)
        if not isinstance(dimension, int) or dimension <= 0:
            raise PineconeVectorStoreError("Pinecone index dimension could not be determined.")
        self._dimension = dimension
        return dimension

    async def upsert(
        self,
        *,
        namespace: str,
        vectors: list[PineconeVectorRecord],
    ) -> None:
        if not vectors:
            return

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            payload = [
                {
                    "id": vector.id,
                    "values": vector.values,
                    "metadata": vector.metadata,
                }
                for vector in batch
            ]
            try:
                await asyncio.to_thread(
                    self._get_index().upsert,
                    vectors=payload,
                    namespace=namespace,
                )
            except PineconeException as exc:
                raise PineconeVectorStoreError(str(exc) or "Pinecone upsert failed.") from exc

    async def query(
        self,
        *,
        namespace: str,
        query_vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[PineconeQueryMatch]:
        try:
            response = await asyncio.to_thread(
                self._get_index().query,
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=metadata_filter,
            )
        except PineconeException as exc:
            raise PineconeVectorStoreError(str(exc) or "Pinecone query failed.") from exc

        matches = getattr(response, "matches", None)
        if not isinstance(matches, list):
            return []

        results: list[PineconeQueryMatch] = []
        for match in matches:
            match_id = getattr(match, "id", "")
            if not isinstance(match_id, str) or not match_id:
                continue

            raw_score = getattr(match, "score", None)
            score = float(raw_score) if isinstance(raw_score, (int, float)) else None
            results.append(
                PineconeQueryMatch(
                    id=match_id,
                    score=score,
                    metadata=_normalize_metadata(getattr(match, "metadata", None)),
                )
            )
        return results

    async def delete_by_filter(
        self,
        *,
        namespace: str,
        metadata_filter: dict[str, Any],
    ) -> None:
        try:
            await asyncio.to_thread(
                self._get_index().delete,
                namespace=namespace,
                filter=metadata_filter,
            )
        except PineconeException as exc:
            raise PineconeVectorStoreError(str(exc) or "Pinecone delete failed.") from exc
