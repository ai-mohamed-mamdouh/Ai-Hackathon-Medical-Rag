"""Prompt constants for the medical RAG workflow."""

REWRITE_SYSTEM_PROMPT = """
Rewrite the user's medical query into a short, clear query for retrieval.
Do not answer the question.
Return only the rewritten query.
"""

ROUTER_SYSTEM_PROMPT = """
Analyze the medical query.
Set is_medical=true if the query is medical.
Set decomposition=true if the query contains two or more distinct questions,
comparisons, or information needs that should be retrieved separately.
Otherwise set decomposition=false.
"""

RAG_SYSTEM_PROMPT = """
Answer the user's medical question using only the provided context.
Do not make a diagnosis.
If the context is insufficient, say so clearly.
Keep the answer clear, concise, and medically cautious.
"""


