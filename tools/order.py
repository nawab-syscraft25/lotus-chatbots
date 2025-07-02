import httpx
from typing import Dict
from memory.memory_store import get_session_memory

ORDER_API_URL = "https://portal.lotuselectronics.com/web-api/user/my_order_list?type=completed"
ORDER_API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

async def get_orders(session_id: str) -> Dict:
    """
    Retrieve the user's completed orders using the auth_token stored in session memory.
    """
    memory = get_session_memory(session_id)
    auth_token = memory.get("auth_token")
    print(f"DEBUG: get_orders for session_id={session_id}, auth_token={auth_token}")
    if not auth_token:
        return {"error": "User not authenticated. Please sign in first."}
    headers = ORDER_API_HEADERS.copy()
    headers["auth-token"] = auth_token
    async with httpx.AsyncClient() as client:
        response = await client.get(ORDER_API_URL, headers=headers)
        return response.json()

get_orders_schema = {
    "name": "get_orders",
    "description": "Retrieve the user's completed orders using their session_id (auth_token must be set in session).",
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"}
        },
        "required": ["session_id"]
    }
}
