import re
import unicodedata
from dataclasses import dataclass

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
