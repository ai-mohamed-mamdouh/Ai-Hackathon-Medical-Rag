import json
import re
from difflib import SequenceMatcher
from pathlib import Path


REQUIRED_FIELDS = {
    "original_query",
    "normalized_query",
    "query_type",
    "query_tags",
    "difficulty",
    "answerable",
    "safety_critical",
    "gold_answer",
    "gold_claims",
    "gold_evidence",
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def query_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def load_json_list(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    return data


def build_chunk_index(chunks: list) -> dict:
    index = {}

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")

        if chunk_id:
            index[chunk_id] = chunk

    return index


def validate_item(item: dict, chunk_index: dict | None = None) -> list[str]:
    errors = []

    if not isinstance(item, dict):
        return ["Item is not a JSON object."]

    missing = REQUIRED_FIELDS - set(item.keys())

    if missing:
        errors.append(
            f"Missing fields: {sorted(missing)}"
        )

    if not str(item.get("original_query", "")).strip():
        errors.append("original_query is empty.")

    if not str(item.get("normalized_query", "")).strip():
        errors.append("normalized_query is empty.")

    if not isinstance(item.get("query_tags"), list):
        errors.append("query_tags must be a list.")

    if not isinstance(item.get("gold_claims"), list):
        errors.append("gold_claims must be a list.")

    answerable = item.get("answerable")

    if not isinstance(answerable, bool):
        errors.append("answerable must be boolean.")

    if not isinstance(item.get("safety_critical"), bool):
        errors.append("safety_critical must be boolean.")

    gold_evidence = item.get("gold_evidence")

    if not isinstance(gold_evidence, list):
        errors.append("gold_evidence must be a list.")
        return errors

    if answerable:
        if not str(item.get("gold_answer", "")).strip():
            errors.append(
                "Answerable query must have gold_answer."
            )

        if not item.get("gold_claims"):
            errors.append(
                "Answerable query must have at least one gold_claim."
            )

        if not gold_evidence:
            errors.append(
                "Answerable query must have gold_evidence."
            )

    has_direct_evidence = False

    for idx, evidence in enumerate(gold_evidence):
        prefix = f"gold_evidence[{idx}]"

        if not isinstance(evidence, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        chunk_id = evidence.get("chunk_id")
        file_id = evidence.get("file_id")
        page_number = evidence.get("page_number")
        relevance = evidence.get("relevance")

        if not chunk_id:
            errors.append(f"{prefix}.chunk_id is empty.")

        if not file_id:
            errors.append(f"{prefix}.file_id is empty.")

        if page_number is None:
            errors.append(
                f"{prefix}.page_number is missing."
            )

        if relevance not in {2, 3}:
            errors.append(
                f"{prefix}.relevance must be 2 or 3."
            )

        if relevance == 3:
            has_direct_evidence = True

        # Validate against real corpus chunks
        if chunk_index is not None and chunk_id:
            source_chunk = chunk_index.get(chunk_id)

            if source_chunk is None:
                errors.append(
                    f"{prefix}: chunk_id not found in corpus."
                )
                continue

            source_file_id = source_chunk.get("file_id")
            source_page = source_chunk.get("page_number")

            if file_id != source_file_id:
                errors.append(
                    f"{prefix}: file_id does not match source chunk."
                )

            if page_number != source_page:
                errors.append(
                    f"{prefix}: page_number does not match source chunk."
                )

    if answerable and gold_evidence and not has_direct_evidence:
        errors.append(
            "Answerable query must contain at least one relevance=3 evidence."
        )

    return errors


def get_evidence_signature(item: dict) -> tuple:
    evidence = item.get("gold_evidence", [])

    return tuple(
        sorted(
            (
                e.get("chunk_id"),
                e.get("file_id"),
                e.get("page_number"),
            )
            for e in evidence
            if isinstance(e, dict)
        )
    )


def is_duplicate(
    item: dict,
    accepted_items: list[dict],
    fuzzy_threshold: float = 0.94,
) -> tuple[bool, str | None]:

    current_original = normalize_text(
        item.get("original_query", "")
    )

    current_normalized = normalize_text(
        item.get("normalized_query", "")
    )

    current_answer = normalize_text(
        item.get("gold_answer", "")
    )

    current_evidence = get_evidence_signature(item)

    for existing in accepted_items:
        existing_original = normalize_text(
            existing.get("original_query", "")
        )

        existing_normalized = normalize_text(
            existing.get("normalized_query", "")
        )

        existing_answer = normalize_text(
            existing.get("gold_answer", "")
        )

        existing_evidence = get_evidence_signature(existing)

        # Exact query duplicate
        if current_original == existing_original:
            return True, "Exact original_query duplicate."

        if current_normalized == existing_normalized:
            return True, "Exact normalized_query duplicate."

        # Same evidence + same answer
        if (
            current_evidence
            and current_evidence == existing_evidence
            and current_answer == existing_answer
        ):
            return True, "Same evidence and gold_answer."

        # Fuzzy query duplicate
        similarity = max(
            query_similarity(
                current_original,
                existing_original,
            ),
            query_similarity(
                current_normalized,
                existing_normalized,
            ),
        )

        if similarity >= fuzzy_threshold:
            return (
                True,
                f"Semantic-like duplicate. Similarity={similarity:.3f}",
            )

    return False, None


def prepare_retrieval_test_dataset(
    questions_path: str,
    output_path: str,
    chunks_path: str | None = None,
    rejected_path: str | None = None,
    fuzzy_threshold: float = 0.94,
    id_prefix: str = "med_",
) -> dict:
    """
    Pipeline:
        Validate Ground Truth
        -> Deduplicate
        -> Assign query_id
        -> Save clean dataset
    """

    questions = load_json_list(questions_path)

    chunk_index = None

    if chunks_path:
        chunks = load_json_list(chunks_path)
        chunk_index = build_chunk_index(chunks)

    accepted = []
    rejected = []

    for source_index, item in enumerate(questions):
        errors = validate_item(
            item=item,
            chunk_index=chunk_index,
        )

        if errors:
            rejected.append(
                {
                    "source_index": source_index,
                    "reason": "validation_failed",
                    "errors": errors,
                    "item": item,
                }
            )
            continue

        duplicate, duplicate_reason = is_duplicate(
            item=item,
            accepted_items=accepted,
            fuzzy_threshold=fuzzy_threshold,
        )

        if duplicate:
            rejected.append(
                {
                    "source_index": source_index,
                    "reason": "duplicate",
                    "errors": [duplicate_reason],
                    "item": item,
                }
            )
            continue

        cleaned_item = dict(item)

        # Remove temporary ID
        cleaned_item["query_id"] = None

        accepted.append(cleaned_item)

    # Assign final IDs after validation + deduplication
    width = max(3, len(str(len(accepted))))

    for index, item in enumerate(accepted, start=1):
        item["query_id"] = (
            f"{id_prefix}{index:0{width}d}"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            accepted,
            f,
            ensure_ascii=False,
            indent=2,
        )

    if rejected_path is None:
        rejected_path = (
            output_path.parent
            / f"{output_path.stem}_rejected.json"
        )
    else:
        rejected_path = Path(rejected_path)

    with open(
        rejected_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rejected,
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "input_queries": len(questions),
        "accepted_queries": len(accepted),
        "rejected_queries": len(rejected),
        "clean_file": str(output_path),
        "rejected_file": str(rejected_path),
    }

    return summary


if __name__ == '__main__' :
    summary = prepare_retrieval_test_dataset(
    questions_path="evaluation/test_dataset/guideline_test_dataset.json",
    chunks_path="giddiness_chunks.json",
    output_path="evaluation/test_dataset/guideline_test_dataset_clean.json",
    )

    print(summary)