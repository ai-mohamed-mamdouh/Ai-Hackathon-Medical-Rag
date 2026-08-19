"""Deterministic source metadata resolution for generated chunk references."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from src.generation.orchestration.medical_rag.schemas import UsedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceResolutionResult:
    """Resolved source metadata and non-fatal source warnings."""

    sources: list[dict[str, Any]]
    warnings: list[str]


def resolve_source_metadata(
    used_chunks: Sequence[UsedChunk],
    document_groups: Sequence[Sequence[Document]],
) -> SourceResolutionResult:
    """Resolve complete metadata using exact file/chunk identifier pairs.

    The first retrieved document for a duplicate identifier pair is
    authoritative. Generated references preserve their original order and are
    deduplicated before lookup.
    """
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    warnings: list[str] = []

    for group_index, document_group in enumerate(document_groups):
        for document_index, document in enumerate(document_group):
            metadata = document.metadata
            if not isinstance(metadata, Mapping):
                warning = (
                    "Retrieved document metadata was not a mapping at "
                    f"group {group_index}, index {document_index}; document skipped."
                )
                warnings.append(warning)
                logger.warning(warning)
                continue

            key = _metadata_key(metadata)
            if key is None:
                warning = (
                    "Retrieved document was missing a valid file_id/chunk_id pair at "
                    f"group {group_index}, index {document_index}; document skipped."
                )
                warnings.append(warning)
                logger.warning(warning)
                continue

            if key in lookup:
                warning = (
                    "Duplicate retrieved source pair "
                    f"({_display_identifier(key[0])}, {_display_identifier(key[1])}) "
                    "was ignored; the first match remains authoritative."
                )
                warnings.append(warning)
                logger.warning(warning)
                continue

            lookup[key] = dict(metadata)

    sources: list[dict[str, Any]] = []
    seen_references: set[tuple[str, str]] = set()
    invalid_reference_count = 0

    for used_chunk in used_chunks:
        key = (str(used_chunk.file_id), str(used_chunk.chunk_id))
        if key in seen_references:
            continue
        seen_references.add(key)

        metadata = lookup.get(key)
        if metadata is None:
            invalid_reference_count += 1
            warning = (
                "Generated source reference did not match a retrieved document: "
                f"file_id={_display_identifier(key[0])}, "
                f"chunk_id={_display_identifier(key[1])}."
            )
            warnings.append(warning)
            logger.warning(warning)
            continue

        sources.append(dict(metadata))

    logger.info(
        "Source metadata resolution completed.",
        extra={
            "used_reference_count": len(used_chunks),
            "valid_source_count": len(sources),
            "invalid_reference_count": invalid_reference_count,
        },
    )
    return SourceResolutionResult(sources=sources, warnings=warnings)


def _metadata_key(metadata: Mapping[str, Any]) -> tuple[str, str] | None:
    """Build an exact source key when both metadata identifiers are present."""
    file_id = metadata.get("file_id")
    chunk_id = metadata.get("chunk_id")

    if file_id is None or chunk_id is None:
        return None

    file_id_text = str(file_id).strip()
    chunk_id_text = str(chunk_id).strip()
    if not file_id_text or not chunk_id_text:
        return None

    return file_id_text, chunk_id_text


def _display_identifier(value: str, limit: int = 64) -> str:
    """Limit identifier length in logs and warning messages."""
    normalized = value.replace("\n", " ").replace("\r", " ")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."
