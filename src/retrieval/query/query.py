import re
import unicodedata
from dataclasses import dataclass

from src.retrieval.query.llm_for_query import decomposetion_chain


@dataclass
class Query:
    original_query: str
    normalized_query: str = ""


class QueryProcessor:
    MEDICAL_TERMS = {
        "t2dm": "T2DM",
        "ckd": "CKD",
        "copd": "COPD",
        "egfr": "eGFR",
        "hba1c": "HbA1c",
        "bmi": "BMI",
        "acei": "ACEi",
        "arb": "ARB",
    }

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = (
            text.replace("–", "-")
            .replace("—", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
        )
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        text = re.sub(r"([!?.,])\1+", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()

        for term, canonical in self.MEDICAL_TERMS.items():
            text = re.sub(
                rf"\b{re.escape(term)}\b",
                canonical,
                text,
                flags=re.IGNORECASE,
            )

        return text

    def normalize_query(self, query: Query) -> Query:
        query.normalized_query = self.normalize_text(query.original_query)
        return query

    def decompose_query(self, query: Query) -> list[Query]:
        result = decomposetion_chain.invoke({
            "query": query.original_query
        })

        queries = []
        for sub_query in result["queries"]:
            normalized_sub_query = self.normalize_text(sub_query)
            queries.append(
                Query(
                    original_query=sub_query,
                    normalized_query=normalized_sub_query,
                )
            )

        return queries


if __name__ == "__main__":
    query = Query(
        original_query="What are the symptoms and treatments of diabetes?"
    )
    result = QueryProcessor().decompose_query(query=query)

    print("result=====================")
    for q in result:
        print(q.original_query)
        print(q.normalized_query)
        print("===========================")
