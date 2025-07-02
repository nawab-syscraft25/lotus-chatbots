import httpx
from typing import Dict
from memory.memory_store import get_session_memory
import json
import re

ORDER_API_URL = "https://portal.lotuselectronics.com/web-api/user/my_order_list?type=completed"
ORDER_API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

async def get_orders(auth_token: str) -> Dict:
    """
    Retrieve the user's completed orders using the auth_token.
    """
    if not auth_token:
        return {"error": "User not authenticated. Please sign in first."}
    headers = ORDER_API_HEADERS.copy()
    headers["Auth_Token"] = auth_token
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(ORDER_API_URL, headers=headers)
            print("DEBUG get_orders API response:", response.text)
            return response.json()
    except Exception as e:
        print("ERROR in get_orders:", str(e))
        return {"error": f"Failed to fetch orders: {str(e)}"}

get_orders_schema = {
    "name": "get_orders",
    "description": "Retrieve the user's completed orders using their auth_token (must be set in session).",
    "parameters": {
        "type": "object",
        "properties": {
            "auth_token": {"type": "string"}
        },
        "required": ["auth_token"]
    }
}

def extract_json_from_response(text):
    try:
        return json.loads(text)
    except Exception as e:
        print("ERROR: Failed to parse JSON from LLM response:", text, "Exception:", e)
        match = re.search(r'({.*})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception as e2:
                print("ERROR: Failed to parse JSON from matched group:", match.group(1), "Exception:", e2)
    return {
        "status": "error",
        "data": {
            "answer": "Sorry, I could not process the response properly.",
            "products": [],
            "end": ""
        }
    }
