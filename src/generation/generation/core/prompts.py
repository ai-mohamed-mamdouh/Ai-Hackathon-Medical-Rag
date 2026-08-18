"""Prompt constants for the medical RAG workflow."""

REWRITE_SYSTEM_PROMPT = """
Rewrite the user's medical query into a short, clear query for retrieval.
Do not answer the question.
Return only the rewritten query.
"""

ROUTER_SYSTEM_PROMPT = """
Analyze the medical query and decide the appropriate retrieval route.
If needed, decompose complex questions into simple searchable sub-queries.
Return only the required routing result.
"""

RAG_SYSTEM_PROMPT = """
Answer the user's medical question using only the provided context.
Do not make a diagnosis.
If the context is insufficient, say so clearly.
Keep the answer clear, concise, and medically cautious.
"""


