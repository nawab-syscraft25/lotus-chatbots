from typing import Dict
from vector_search import search_vector_db_async

vector_search_schema = {
    "name": "vector_search_products",
    "description": "Semantic search for products using vector database.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}

async def vector_search_products(query: str) -> Dict:
    """Semantic product search using vector DB."""
    results = await search_vector_db_async(query)
    return results 