"""Retrieval query model.

Keep the repository's existing implementation when it already provides the
same constructor and attributes.
"""

from pydantic import BaseModel, ConfigDict, field_validator


class Query(BaseModel):
    """A user query and its normalized representation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    original_query: str
    normalized_query: str = ""

    @field_validator("original_query")
    @classmethod
    def validate_original_query(cls, value: str) -> str:
        """Require a non-empty original query without changing its contents."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("original_query must be a non-empty string.")
        return value
