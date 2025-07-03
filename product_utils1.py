import requests
import re
from typing import Dict, Optional, Tuple

API_URL = "https://portal.lotuselectronics.com/web-api/home/product_detail"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "content-type": "application/x-www-form-urlencoded",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.0.0.0"
}
PRODUCT_ID_PATTERN = re.compile(r"/(\d+)/?$")
REQUEST_TIMEOUT = 10  # seconds

def extract_product_id_from_url(url: str) -> Optional[str]:
    match = PRODUCT_ID_PATTERN.search(url)
    return match.group(1) if match else None

def get_product_details(product_id: str) -> Optional[Dict]:
    data = {
        "product_id": product_id,
        "cat_name": f"/product/{product_id}",
        "product_name": f"product-{product_id}"
    }
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        return result.get("data", {}).get("product_detail")
    except Exception:
        return None

def get_first_image(product_detail: Dict) -> Optional[str]:
    for field in ["product_image", "product_images_350"]:
        val = product_detail.get(field)
        if isinstance(val, list) and val:
            return val[0]
        elif isinstance(val, str) and val:
            return val
    return None

def get_product_stock_status(product_link: str) -> Tuple[bool, Optional[str], Optional[str]]:
    product_id = extract_product_id_from_url(product_link)
    if not product_id:
        return False, None, "Invalid product URL format"
    detail = get_product_details(product_id)
    if not detail:
        return False, None, "Product details not found"
    instock = detail.get("instock", "").lower() == "yes"
    out_of_stock = detail.get("out_of_stock", "0") == "0"
    quantity = int(detail.get("product_quantity", "0")) > 0
    is_in_stock = instock and out_of_stock and quantity
    image = get_first_image(detail)
    return is_in_stock, image, None