import httpx
from typing import Optional

API_URL = "https://portal.lotuselectronics.com/web-api/home/check_category"
API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "content-type": "application/json",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
}

async def check_category(category: str, maincat: Optional[str] = None) -> dict:
    payload = {
        "category": category,
        "maincat": maincat or category
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json=payload, headers=API_HEADERS)
        return response.json()

check_category_schema = {
    "name": "check_category",
    "description": "Check if a category exists in Lotus Electronics.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Category to check (required)"},
            "maincat": {"type": "string", "description": "Main category (optional)"}
        },
        "required": ["category"]
    }
} 