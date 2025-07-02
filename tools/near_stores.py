import httpx

STORES_URL = "https://portal.lotuselectronics.com/web-api/home/stores"
STORES_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

async def check_near_stores(pin_code: str, auth_token: str = None) -> dict:
    headers = STORES_HEADERS.copy()
    if auth_token:
        headers["auth-token"] = auth_token
    data = {
        "pin_code": pin_code
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(STORES_URL, data=data, headers=headers)
        return response.json()

check_near_stores_schema = {
    "name": "check_near_stores",
    "description": "Check for nearby Lotus Electronics stores by pin code (zip code).",
    "parameters": {
        "type": "object",
        "properties": {
            "pin_code": {"type": "string", "description": "The user's pin code (zip code)."},
            "auth_token": {"type": "string", "description": "User's auth token if available.", "default": None}
        },
        "required": ["pin_code"]
    }
} 