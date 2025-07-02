import httpx

OFFERS_URL = "https://portal.lotuselectronics.com/web-api/cat_page_filter/offer_slider"
OFFERS_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "content-type": "application/json",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

async def get_current_offers(page: str = "home", ctp: int = 0) -> dict:
    payload = {"page": page, "ctp": ctp}
    async with httpx.AsyncClient() as client:
        response = await client.post(OFFERS_URL, json=payload, headers=OFFERS_HEADERS)
        return response.json()

get_current_offers_schema = {
    "name": "get_current_offers",
    "description": "Fetch the current promotional offers from Lotus Electronics.",
    "parameters": {
        "type": "object",
        "properties": {
            "page": {"type": "string", "default": "home", "description": "Page to fetch offers for (default: home)"},
            "ctp": {"type": "integer", "default": 0, "description": "CTP value (default: 0)"}
        },
        "required": []
    }
} 