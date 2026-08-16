from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config.settings import settings
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model=settings.QUERY_MODEL_NAME,
    temperature=0
)
# Query ReWrite

# Decomposetion

decomposetion_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a query decomposition assistant for a RAG system.
Decompose the user's query into small, independent search queries.
Rules:
- Preserve the original meaning.
- Each query should represent one clear information need.
- Do not add information that the user did not ask for.
- If the query is already simple, return it as a single query.
- Return JSON only.

Format:
{{
    "queries": [
        "query 1",
        "query 2"
    ]
}}
"""
    ),
    ("user", "{query}")
])

parser = JsonOutputParser()

decomposetion_chain = decomposetion_prompt | llm | parser