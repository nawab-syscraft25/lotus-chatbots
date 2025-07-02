import os
import re
import json
import logging
import asyncio
from typing import Dict, List, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOTUS_API_BASE = "https://portal.lotuselectronics.com/web-api/home"
LOTUS_API_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "auth-key": "Web2@!9",
    "auth-token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiNjg5MzYiLCJpYXQiOjE3NDg5NDc2NDEsImV4cCI6MTc0ODk2NTY0MX0.uZeQseqc6mpm5vkOAmEDgUeWIfOI5i_FnHJRaUBWlMY",
    "content-type": "application/x-www-form-urlencoded",
    "end-client": "Lotus-Web",
    "origin": "https://www.lotuselectronics.com",
    "referer": "https://www.lotuselectronics.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0"
}

async_client = httpx.AsyncClient(timeout=10.0)
PRODUCT_PROCESS_LIMIT = 4

def extract_product_category_for_api(query: str) -> str:
    # ... (same as provided)
    query_lower = query.lower()
    price_patterns = [
        r'₹\d+', r'\d+\s*rs?', r'budget', r'price', r'cost', r'under', r'above', 
        r'between', r'around', r'approximately', r'best deals', r'deals', r'offer',
        r'my budget is', r'budget of', r'budget for', r'within budget', r'show me the price of',
        r'what is the price of', r'how much is', r'how much does', r'cost of'
    ]
    for pattern in price_patterns:
        query_lower = re.sub(pattern, '', query_lower)
    words = query_lower.split()
    has_numbers = any(re.search(r'\d+', word) for word in words)
    has_model_indicators = any(word in ['pro', 'max', 'ultra', 'plus', 'mini', 'se', 'xl'] for word in words)
    has_brand_indicators = any(word in ['iphone', 'samsung', 'galaxy', 'vivo', 'oppo', 'oneplus', 'xiaomi', 'realme', 'pixel', 'sony', 'lg'] for word in words)
    if (len(words) >= 3 and (has_numbers or has_model_indicators)) or has_brand_indicators:
        product_name = ' '.join([word for word in words if len(word) > 1])
        return product_name.strip()
    category_mappings = {
        'tv': ['tv', 'television', 'televisions', 'smart tv', 'led tv', 'oled tv', 'qled tv', '4k tv', 'ultra hd tv'],
        'smartphone': ['smartphone', 'smartphones', 'mobile', 'mobiles', 'phone', 'phones', 'android phone'],
        'laptop': ['laptop', 'laptops', 'notebook', 'notebooks', 'computer', 'pc'],
        'ac': ['ac', 'air conditioner', 'air conditioners', 'split ac', 'window ac', 'cooling'],
        'refrigerator': ['refrigerator', 'refrigerators', 'fridge', 'fridges', 'cooling'],
        'washing machine': ['washing machine', 'washing machines', 'washer', 'laundry'],
        'microwave': ['microwave', 'microwaves', 'oven', 'cooking'],
        'headphones': ['headphones', 'headphone', 'earphones', 'earphone', 'earbuds'],
        'speaker': ['speaker', 'speakers', 'bluetooth speaker', 'sound'],
        'camera': ['camera', 'cameras', 'digital camera', 'photography'],
        'tablet': ['tablet', 'tablets', 'ipad', 'android tablet'],
        'printer': ['printer', 'printers', 'printing'],
        'monitor': ['monitor', 'monitors', 'computer monitor', 'display'],
        'keyboard': ['keyboard', 'keyboards', 'typing'],
        'mouse': ['mouse', 'mice', 'pointing'],
        'router': ['router', 'routers', 'wifi router', 'internet'],
        'power bank': ['power bank', 'power banks', 'powerbank', 'battery'],
        'charger': ['charger', 'chargers', 'mobile charger', 'charging'],
        'cable': ['cable', 'cables', 'usb cable', 'hdmi cable', 'wire'],
        'adapter': ['adapter', 'adapters', 'power adapter', 'connector']
    }
    for category, keywords in category_mappings.items():
        for keyword in keywords:
            if keyword in query_lower:
                return category
    if words:
        first_word = words[0]
        for category, keywords in category_mappings.items():
            if first_word in keywords:
                return category
        for word in words:
            if len(word) > 2 and word not in ['the', 'and', 'for', 'with', 'best', 'good', 'new', 'my', 'is', 'are', 'was', 'were']:
                return word
    return query

async def get_product_details(product_id: str) -> Tuple[bool, Dict]:
    try:
        url = f"{LOTUS_API_BASE}/product_detail"
        data = {
            "product_id": product_id,
            "cat_name": f"/product/{product_id}",
            "product_name": f"product-{product_id}"
        }
        response = await async_client.post(url, headers=LOTUS_API_HEADERS, data=data)
        response.raise_for_status()
        result = response.json()
        if "data" in result and "product_detail" in result["data"]:
            detail = result["data"]["product_detail"]
            instock = detail.get("instock", "").lower()
            out_of_stock = detail.get("out_of_stock", "0")
            quantity = int(detail.get("product_quantity", "0"))
            is_in_stock = (instock == "yes" and out_of_stock == "0" and quantity > 0)
            return is_in_stock, detail
        return False, {}
    except Exception as e:
        logger.error(f"Product detail error: {str(e)}")
        return False, {}

async def search_lotus_products(query: str, limit: int = 10) -> List[Dict]:
    try:
        url = f"{LOTUS_API_BASE}/search_products"
        data = {
            "search_text": query.strip(),
            "alias": "",
            "is_brand_search": "0",
            "limit": str(limit),
            "offset": "0",
            "orderby": ""
        }
        response = await async_client.post(url, headers=LOTUS_API_HEADERS, data=data)
        response.raise_for_status()
        result = response.json()
        
        # Handle both dict and list responses from the API
        data = result.get("data", {})
        if isinstance(data, dict):
            products = data.get("products", [])
        elif isinstance(data, list):
            products = data
        else:
            products = []
            
        if not products:
            return []
        products = products[:PRODUCT_PROCESS_LIMIT]
        tasks = [get_product_details(p["product_id"]) for p in products if "product_id" in p]
        details_results = await asyncio.gather(*tasks)
        processed_products = []
        for idx, (is_in_stock, product_detail) in enumerate(details_results):
            if not product_detail:
                continue
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
            processed_products.append({
                "name": product_detail.get("product_name", ""),
                "link": f"https://www.lotuselectronics.com/product/{product_detail.get('uri_slug', '')}/{product_detail.get('product_id', '')}",
                "price": f"₹{product_detail.get('product_mrp', 'N/A')}",
                "image": product_detail.get("product_image", [""])[0] if isinstance(product_detail.get("product_image"), list) else product_detail.get("product_image", ""),
                "brand": product_detail.get("brand_name", "N/A"),
                "in_stock": product_detail.get("instock", "").lower() == "yes",
                "stock_status": "" if product_detail.get("instock", "").lower() == "yes" else "Out of Stock",
                "features": features,
                "score": 0.0,
                "product_sku" : product_detail.get("product_sku", 'N/A'),
                "product_id" : product_detail.get("product_id", 'N/A')
            })
        return processed_products
    except Exception as e:
        logger.error(f"API search error: {str(e)}")
        return []

def extract_json_from_string(text: str) -> Dict:
    try:
        json_str = re.sub(r"```json|```", "", text).strip()
        if not json_str.startswith('{'):
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx + 1]
        result = json.loads(json_str)
        print(f"✅ JSON parsed successfully")
        return result
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {str(e)}")
        print(f"🔍 Raw response: {text[:300]}...")
        return {}
    except Exception as e:
        print(f"❌ Unexpected error parsing JSON: {str(e)}")
        print(f"🔍 Raw response: {text[:300]}...")
        return {}

# This is the function to be called by the agent
async def search_products(query: str) -> dict:
    """Search for products using Lotus Electronics API."""
    products = await search_lotus_products(query)
    return {"results": products}

search_products_schema = {
    "name": "search_products",
    "description": "Search for products in the catalog.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}
