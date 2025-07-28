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
    """
    Fetch in-stock products by category from Lotus Electronics.
    """
    try:
        params = {
            "category": category,
            "subcategory": subcategory or "",
            "queryStr": queryStr or "",
            "orderby": orderby or "",
            "city": city or "null",
            "pincode": pincode or ""
        }
        
        print(f"Fetching products with params: {params}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(API_URL, params=params, headers=API_HEADERS)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
        print(f"API Response status: {response.status_code}")
        
        # Extract in-store products
        instore = data.get("data", {}).get("cat_detail", {}).get("instore", [])
        if not isinstance(instore, list):
            print(f"Warning: instore is not a list, got: {type(instore)}")
            instore = []
        
        print(f"Found {len(instore)} total products")
        
        # Filter for in-stock products only
        in_stock = []
        for prod in instore:
            if not isinstance(prod, dict):
                continue
            
            # Check if product has quantity > 0
            quantity = prod.get("product_quantity", "0")
            try:
                if int(str(quantity)) > 0:
                    in_stock.append(prod)
            except (ValueError, TypeError):
                # If quantity can't be converted to int, skip
                continue
        
        print(f"Found {len(in_stock)} in-stock products")
        
        if not in_stock:
            return []
        
        # Extract product IDs
        product_ids = []
        for prod in in_stock:
            if "product_id" in prod and prod["product_id"]:
                product_ids.append(prod["product_id"])
        
        print(f"Fetching details for {len(product_ids)} products")
        
        # Fetch full product details for each product_id
        if not product_ids:
            return []
        
        # Limit concurrent requests to avoid overwhelming the API
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def fetch_with_semaphore(pid):
            async with semaphore:
                return await get_product_details(pid)
        
        tasks = [fetch_with_semaphore(pid) for pid in product_ids]
        details_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        products = []
        for i, result in enumerate(details_results):
            if isinstance(result, Exception):
                print(f"Error fetching details for product {product_ids[i]}: {result}")
                continue
            
            is_in_stock, product_detail = result
            if not product_detail:
                continue
            
            # Extract features (up to 6, as in search.py)
            features = product_detail.get("product_specification", [])
            feature_strings = []
            
            if isinstance(features, list):
                for feature in features[:6]:
                    if isinstance(feature, dict):
                        if 'fkey' in feature and 'fvalue' in feature:
                            feature_strings.append(f"{feature['fkey']}: {feature['fvalue']}")
                        elif 'key' in feature and 'value' in feature:
                            feature_strings.append(f"{feature['key']}: {feature['value']}")
                    elif isinstance(feature, str):
                        feature_strings.append(feature)
            
            # Get product image
            product_image = product_detail.get("product_image", "")
            if isinstance(product_image, list) and product_image:
                product_image = product_image[0]
            elif not isinstance(product_image, str):
                product_image = ""
            
            # Get product price - handle different price fields
            product_mrp = product_detail.get('product_mrp', 'N/A')
            if product_mrp == 'N/A':
                product_mrp = product_detail.get('price', 'N/A')
            if product_mrp == 'N/A':
                product_mrp = product_detail.get('product_price', 'N/A')
            
            # Build the product dict
            product_dict = {
                "name": product_detail.get("product_name", "Product"),
                "product_name": product_detail.get("product_name", "Product"),
                "link": f"https://www.lotuselectronics.com/product/{product_detail.get('uri_slug', '')}/{product_detail.get('product_id', '')}",
                "price": f"₹{product_mrp}",
                "product_mrp": product_mrp,  # Keep original price for filtering
                "image": product_image,
                "brand": product_detail.get("brand_name", "N/A"),
                "in_stock": product_detail.get("instock", "").lower() == "yes",
                "stock_status": "" if product_detail.get("instock", "").lower() == "yes" else "Out of Stock",
                "features": feature_strings,
                "product_sku": product_detail.get("product_sku", 'N/A'),
                "product_id": product_detail.get("product_id", 'N/A')
            }
            
            products.append(product_dict)
        
        print(f"Successfully processed {len(products)} products")
        return products
        
    except httpx.HTTPStatusError as e:
        print(f"HTTP error in get_products_by_category: {e}")
        return []
    except httpx.RequestError as e:
        print(f"Request error in get_products_by_category: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in get_products_by_category: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []


def filter_products_by_price(products, price_filter):
    """
    Helper function to filter products by price range.
    """
    if not price_filter or not products:
        return products
    
    filtered = []
    for prod in products:
        # Get price from multiple possible fields
        price_str = prod.get('product_mrp') or prod.get('price') or prod.get('product_price', '')
        price = parse_price(price_str)
        
        if price is None:
            continue
        
        in_range = True
        if "$lte" in price_filter and price > price_filter["$lte"]:
            in_range = False
        if "$gte" in price_filter and price < price_filter["$gte"]:
            in_range = False
        
        if in_range:
            filtered.append(prod)
    
    print(f"Filtered {len(filtered)} products from {len(products)} based on price range")
    return filtered


def parse_price(price_str):
    """
    Parse price string to float, handling various formats.
    """
    if not price_str or price_str in ['N/A', 'n/a', '']:
        return None
    
    try:
        # Remove currency symbols and commas
        clean_price = str(price_str).replace('₹', '').replace(',', '').strip()
        return float(clean_price)
    except (ValueError, TypeError):
        return None


# Tool schema for OpenAI function calling
get_products_by_category_schema = {
    "name": "get_products_by_category",
    "description": "Fetch in-stock products by category from Lotus Electronics. This tool should be used when users ask for products in specific categories or subcategories.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string", 
                "description": "Main category slug (required) - e.g., 'mobile-phones', 'laptops', 'televisions'"
            },
            "subcategory": {
                "type": "string", 
                "description": "Subcategory slug (optional) - e.g., 'smartphones', 'gaming-laptops'"
            },
            "queryStr": {
                "type": "string", 
                "description": "Query string for filters (optional) - e.g., 'price=[1000,50000]'"
            },
            "orderby": {
                "type": "string", 
                "description": "Order by field (optional) - e.g., 'price_asc', 'price_desc', 'name'"
            },
            "city": {
                "type": "string", 
                "description": "City for location-based results (optional)"
            },
            "pincode": {
                "type": "string", 
                "description": "Pincode for location-based results (optional)"
            }
        },
        "required": ["category"]
    }
}