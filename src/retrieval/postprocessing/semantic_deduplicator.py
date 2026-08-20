import re

import numpy as np
from langchain_core.documents import Document


class SemanticDeduplicator:
    def __init__(
        self,
        embeddings,
        similarity_threshold: float = 0.88,
        lexical_threshold: float = 0.60,
    ):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.lexical_threshold = lexical_threshold

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(
            re.findall(
                r"\b\w+\b",
                text.lower(),
            )
        )

    @classmethod
    def _lexical_similarity(
        cls,
        text_a: str,
        text_b: str,
    ) -> float:
        tokens_a = cls._tokenize(text_a)
        tokens_b = cls._tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union)

    def deduplicate(
        self,
        documents: list[Document],
    ) -> list[Document]:
        if len(documents) <= 1:
            return documents

        texts = [doc.page_content for doc in documents]

        vectors = np.asarray(
            self.embeddings.embed_documents(texts),
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        vectors = vectors / np.clip(
            norms,
            1e-12,
            None,
        )

        selected_indices: list[int] = []

        for index, vector in enumerate(vectors):
            if not selected_indices:
                documents[index].metadata[
                    "semantic_max_similarity"
                ] = 0.0
                documents[index].metadata[
                    "lexical_max_similarity"
                ] = 0.0

                selected_indices.append(index)
                continue

            is_duplicate = False
            max_semantic = 0.0
            max_lexical = 0.0

            for selected_index in selected_indices:
                semantic_similarity = float(
                    vectors[selected_index] @ vector
                )

                lexical_similarity = self._lexical_similarity(
                    texts[selected_index],
                    texts[index],
                )

                max_semantic = max(
                    max_semantic,
                    semantic_similarity,
                )

                max_lexical = max(
                    max_lexical,
                    lexical_similarity,
                )

                if (
                    semantic_similarity
                    >= self.similarity_threshold
                    and lexical_similarity
                    >= self.lexical_threshold
                ):
                    is_duplicate = True
                    break

            documents[index].metadata[
                "semantic_max_similarity"
            ] = max_semantic

            documents[index].metadata[
                "lexical_max_similarity"
            ] = max_lexical

            if not is_duplicate:
                selected_indices.append(index)

        return [
            documents[index]
            for index in selected_indices
        ]