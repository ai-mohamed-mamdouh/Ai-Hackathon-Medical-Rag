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

    def normalize_query(self, query: Query) -> Query:
        text = query.original_query

        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Normalize special punctuation
        text = (
            text
            .replace("–", "-")
            .replace("—", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
        )

        # Remove invisible characters
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

        # Remove duplicated punctuation
        text = re.sub(r"([!?.,])\1+", r"\1", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Normalize known medical terms
        for term, canonical in self.MEDICAL_TERMS.items():
            text = re.sub(
                rf"\b{re.escape(term)}\b",
                canonical,
                text,
                flags=re.IGNORECASE,
            )

        query.normalized_query = text

        return query

    def decompose_query(self, query: Query) -> list[Query]:
        result = decomposetion_chain.invoke({
            "query": query.original_query
        })

        return [
            Query(
                original_query=query.original_query,
                normalized_query=sub_query
            )
            for sub_query in result["queries"]
        ]

if __name__ == '__main__' :
    query = Query(
        original_query="What are the symptoms and treatments of diabetes?"
    )
    result = QueryProcessor().decompose_query(query=query)

    print('reslut=====================')
    for q in result :
        print(q.original_query)
        print(q.normalized_query)
        print('===========================')
    