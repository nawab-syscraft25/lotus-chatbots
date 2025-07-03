import httpx
from typing import Optional
import asyncio
from .search import get_product_details

API_URL = "https://portal.lotuselectronics.com/web-api/cat_page_filter/get_products_by_category"
API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
}

async def get_products_by_category(
    category: str,
    subcategory: Optional[str] = None,
    queryStr: Optional[str] = None,
    orderby: Optional[str] = None,
    city: Optional[str] = None,
    pincode: Optional[str] = None
) -> list:
    params = {
        "category": category,
        "subcategory": subcategory or "",
        "queryStr": queryStr or "",
        "orderby": orderby or "",
        "city": city or "null",
        "pincode": pincode or ""
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(API_URL, params=params, headers=API_HEADERS)
        data = response.json()
        # print(data)
        # print("Raw instore:", data.get("data", {}).get("cat_detail", {}).get("instore", []))
        # Only return in-stock products
        instore = data.get("data", {}).get("cat_detail", {}).get("instore", [])
        if not isinstance(instore, list):
            instore = []
        in_stock = [prod for prod in instore if isinstance(prod, dict) and str(prod.get("product_quantity", "0")) != "0"]
        product_ids = [prod["product_id"] for prod in in_stock if "product_id" in prod]
        # Fetch full product details for each product_id
        tasks = [get_product_details(pid) for pid in product_ids]
        details_results = await asyncio.gather(*tasks)
        products = []
        for is_in_stock, product_detail in details_results:
            if not product_detail:
                continue
            # Extract features (up to 6, as in search.py)
            features = product_detail.get("product_specification", [])
            if isinstance(features, list):
                feature_strings = []
                for feature in features[:6]:
                    if isinstance(feature, dict):
                        if 'fkey' in feature and 'fvalue' in feature:
                            feature_strings.append(f"{feature['fkey']}: {feature['fvalue']}")
                        elif 'key' in feature and 'value' in feature:
                            feature_strings.append(f"{feature['key']}: {feature['value']}")
                    elif isinstance(feature, str):
                        feature_strings.append(feature)
                features = feature_strings
            # Build the product dict
            products.append({
                "name": product_detail.get("product_name", "Product"),
                "product_name": product_detail.get("product_name", "Product"),
                "link": f"https://www.lotuselectronics.com/product/{product_detail.get('uri_slug', '')}/{product_detail.get('product_id', '')}",
                "price": f"₹{product_detail.get('product_mrp', 'N/A')}",
                "image": product_detail.get("product_image", [""])[0] if isinstance(product_detail.get("product_image"), list) else product_detail.get("product_image", ""),
                "brand": product_detail.get("brand_name", "N/A"),
                "in_stock": product_detail.get("instock", "").lower() == "yes",
                "stock_status": "" if product_detail.get("instock", "").lower() == "yes" else "Out of Stock",
                "features": features,
                "product_sku": product_detail.get("product_sku", 'N/A'),
                "product_id": product_detail.get("product_id", 'N/A')
            })
        return products

get_products_by_category_schema = {
    "name": "get_products_by_category",
    "description": "Fetch in-stock products by category from Lotus Electronics.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Main category (required)"},
            "subcategory": {"type": "string", "description": "Subcategory (optional)"},
            "queryStr": {"type": "string", "description": "Query string for filters (optional)"},
            "orderby": {"type": "string", "description": "Order by (optional)"},
            "city": {"type": "string", "description": "City (optional)"},
            "pincode": {"type": "string", "description": "Pincode (optional)"}
        },
        "required": ["category"]
    }
} 

