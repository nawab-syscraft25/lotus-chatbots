import httpx

DELIVERY_URL = "https://portal.lotuselectronics.com/web-api/home/delivery_opt"
DELIVERY_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

async def check_product_delivery(product_sku: str, pin_code: str, auth_token: str = None) -> dict:
    headers = DELIVERY_HEADERS.copy()
    if auth_token:
        headers["auth-token"] = auth_token
    data = {
        "itemcode": product_sku,
        "pin_code": pin_code
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(DELIVERY_URL, data=data, headers=headers)
        return response.json()

check_product_delivery_schema = {
    "name": "check_product_delivery",
    "description": "Check if a product can be delivered to a given pin code (zip code). Requires product_sku and pin_code.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_sku": {"type": "string", "description": "The SKU or item code of the product."},
            "pin_code": {"type": "string", "description": "The delivery pin code (zip code)."},
            "auth_token": {"type": "string", "description": "User's auth token if available.", "default": None}
        },
        "required": ["product_sku", "pin_code"]
    }
} 