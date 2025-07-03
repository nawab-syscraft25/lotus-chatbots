# openai_agent.py

import os
import openai
import asyncio
from tools.tool_registry import tool_registry
import re
import json
import traceback
import sys
sys.path.append('.')
from tools.check_category import check_category
from tools.get_products_by_category import get_products_by_category
from vector_search import extract_price_filter, parse_price

openai.api_key = os.getenv("OPENAI_API_KEY")

LOTUS_SYSTEM_PROMPT = (
    "You are Lotus, the official AI assistant for Lotus Electronics. "
    "You help users with product questions, store information, order status, and shopping assistance for Lotus Electronics. "
    "Always answer as a helpful, knowledgeable, and friendly Lotus Electronics representative. "
    "If a user asks about products, use the available tools to search the Lotus Electronics catalog. "
    "Do not mention other retailers or suggest shopping elsewhere. "
    "Do not give the response in Markdown format. "
    "We have Multiple tools. Use the tool when it is required. And Select the tool based on the user's query."
    "If the price is mentioned in the user's query, then use the get_products_by_category tool. But first get the category from the user's query and check the category is valid or not. "
    "Also Handel the price if user say under 15k or above 15k or in 15k so that mean the price is 15000"
    "For Normal Search queary User search_products tool for general search user vector_search_products tool"
    "Like What's the latest price of iphone then use search_products tool. It uses API for Latest info."
    "Instructions: "
    "- Output strict JSON with keys: 'answer', 'products', 'end', and optionally 'comparison' if the user requests a comparison or if it would help them decide. "
    "- If a comparison is included, provide a clear, concise comparison of up to 3 products, focusing on features, price, and unique selling points. "
    "- Include up to 3–4 products with name, link, price (₹), image, 3–4 features. "
    "- Tailor 'answer' to query. "
    "- If no exact product match, suggest closest available alternatives and mention stock status if helpful. "
    "- Skip incomplete items. "
    "- Be very intelligent in product selection and recommendations. Don't show same products again. "
    "- In features always try to show the unique features of the product then other features. "
    "- No markdown, no extra text. JSON only. "
    "- Always try to infer what the customer is looking for, such as their budget, room size, preferred brand, or specific features. "
    "- If the user's request is ambiguous or incomplete, ask clarifying questions to better understand their needs before making recommendations. "
    "- After providing recommendations, ask a relevant follow-up question to help the user make a decision or clarify their needs. "
    "- If the user provides a phone number and it is valid, automatically proceed to send an OTP (or sign in if password is provided), and clearly guide the user to the next authentication step. Do not just thank the user for providing their phone number. "
    "- Never say an OTP has been sent unless you have actually called the send_otp tool and received a successful response. "
    "- Always use the send_otp tool to send an OTP before telling the user that an OTP has been sent. "
    "- If the user requests, or if it would help, offer to compare products and include a 'comparison' key in the JSON response. "
    "- If the user is authenticated (auth_token is present in session), and they ask to view their orders, always use the get_orders tool to fetch and display their orders. "
    "- Do not ask the user to log in or provide their password again if they are already authenticated. "
    "- Only prompt for login or OTP if the user is not authenticated. "
    "- After authentication, use the auth_token for all further tool calls that require authentication. "
    "- If user is asking about the comparison, then always give the comparison. "
    "If No Products and No Comparison, Then Do not give End Message Just Give Answer."
    "Guide the user through the buying process as a helpful Lotus Electronics representative. "
    "If user is asking about the Mobile Phone the use Smartphones as the product."
    "Respond with: "
    "{"
    "  \"status\": \"success\","
    "  \"data\": {"
    "    \"answer\": \"...\","
    "    \"products\": [{\"name\": \"...\", \"link\": \"...\", \"price\": \"₹...\", \"image\": \"...\", \"features\": [\"...\"]}],"
    "    \"comparison\": [{\"name\": \"...\", \"vs_name\": \"...\", \"differences\": [\"...\"]}],"
    "    \"end\": \"...\""
    "  }"
    "}"
)

# Comprehensive category and subcategory mapping
CATEGORY_SUBCATEGORY_MAP = {
    "led-television": [
        "smart-tv","qled-tv","oled-tv","4k-ultra-hd-tv", "8k-ultra-hd-tv", "full-hd-led-tv", "hd-led-tv"
    ],
    "mobile-phone-tablet": [
        "mobile-phones", "smart-wearables-fitness-band", "tablets-ipads", "corded-telephone", "mobile-accessories", 
    ],
    "mobile-phones": [
        "feature-phones", "iphones", "smartphones"
    ],
    "smart-wearables-fitness-band": [
        "smart-watch"
    ],
    "computers-laptops": [
        "computer-accessories", "desktop", "laptops", "printers"
    ],
    "laptops": [
        "convertible-laptop","windows-laptop", "gaming-laptop", "macbook-laptop", "thin-light-laptop"
    ],
    "desktop": [
        "aio-desktop", "tower-desktop"
    ],
    "computer-accessories": [
        "keyboard", "mouse", "pendrives"
    ],
    "printers": [
        "colour-laser-printer", "ink-tank-printer", "inkjet-printer", "laser-printer"
    ],
    "audio-video-home-entertainment": [
        "audio-accessories", "gaming", "gaming-software", "headphones", "home-theatre", "projectors", "sound-bar", "speakers"
    ],
    "gaming": [
        "gaming-console", "gaming-software"
    ],
    "speakers": [
        "amplified-speaker", "bluetooth-speaker", "smart-speaker", "tower-speaker"
    ],
    "headphones": [
        "earphone", "headphone", "neckband", "wireless-headphone"
    ],
    "home-appliances": [
        "air-conditioners", "deep-freezer", "dishwasher", "electronic-safe", "fans", "floor-home-care", "irons-garment-steamer", "refrigerators", "room-coolers", "room-heaters", "sewing-machines", "washing-machines", "water-heaters", "water-purifiers-dispensers"
    ],
    "air-conditioners": [
        "cassette-ac", "portable-ac", "tower-ac", "wall-mounted-split-ac", "window-ac"
    ],
    "fans": [
        "ceiling-fan", "exhaust-fan", "pedestal-fan", "table-fan", "tower-fan", "wall-fan"
    ],
    "refrigerators": [
        "bottom-freezer-refrigerator", "built-in-refrigerator", "double-door-refrigerator", "french-door-refrigerator", "mini-refrigerator", "side-by-side-refrigerator", "single-door-refrigerator", "triple-door-refrigerator"
    ],
    "washing-machines": [
        "cloth-dryer", "front-load-washing-machine", "semi-automatic-washing-machine", "top-load-washing-machine"
    ],
    "water-purifiers-dispensers": [
        "water-cooler", "water-dispenser", "water-purifier"
    ],
    "irons-garment-steamer": [
        "dry-irons", "garment-steamer", "steam-irons"
    ],
    "floor-home-care": [
        "air-purifier", "vacuum-cleaner", "vegetable-fruit-purifier"
    ],
    "electronic-safe": [
        "locker-biometric", "locker-digital", "locker-key-lock"
    ],
    "kitchen-appliances": [
        "attamakers", "beverage-makers", "chimneys", "cooking-appliances", "food-processor", "juicer-mixer-grinder", "microwave-ovens", "sandwich-maker", "toasters-grillers"
    ],
    "beverage-makers": [
        "electric-kettle", "tea-coffee-maker"
    ],
    "cooking-appliances": [
        "air-fryer", "built-in-hob", "built-in-oven", "cooking-hob", "gas-stove", "induction-cooker", "otg", "pop-up-toaster", "rice-induction-cookers"
    ],
    "food-processor": [
        "chopper", "hand-blender"
    ],
    "juicer-mixer-grinder": [
        "hand-mixers", "juicer-mixer", "juicers", "mixer-grinder"
    ],
    "microwave-ovens": [
        "built-in-microwave", "convection-microwave-oven", "grill-microwave", "solo-microwave"
    ],
    "toasters-grillers": [
        "otg", "pop-up-toaster", "sandwich-maker"
    ],
    "cameras": [
        "dslr-cameras"
    ],
    "personal-care": [
        "hair-grooming", "hair-styling"
    ],
    "hair-styling": [
        "hair-brush", "hair-curler", "hair-dryer", "hair-multi-stiller", "hair-straightener", "hair-styler"
    ],
    "hair-grooming": [
        "beard-trimmer", "body-groomer", "hair-brush", "hair-curler", "hair-dryer", "hair-grooming-kit", "hair-multi-stiller", "hair-straightener", "hair-styler", "multi-groomer"
    ],
    "corded-telephone": [
        "cordless-phones", "landline-phones"
    ]
}

def extract_category_and_subcategory(user_query):
    query = user_query.lower().replace('&', 'and').replace('  ', ' ').strip()
    cat_match = None
    subcat_match = None
    # Try to match subcategory first for specificity
    for cat, subcats in CATEGORY_SUBCATEGORY_MAP.items():
        for subcat in subcats:
            if subcat.replace('-', ' ') in query or subcat in query:
                cat_match = cat
                subcat_match = subcat
                return cat_match, subcat_match
    # If no subcategory match, try to match category
    for cat in CATEGORY_SUBCATEGORY_MAP:
        if cat.replace('-', ' ') in query or cat in query:
            cat_match = cat
            return cat_match, None
    # Fallback: try partial match for subcategory
    for cat, subcats in CATEGORY_SUBCATEGORY_MAP.items():
        for subcat in subcats:
            if any(word in subcat for word in query.split()):
                cat_match = cat
                subcat_match = subcat
                return cat_match, subcat_match
    # Fallback: try partial match for category
    for cat in CATEGORY_SUBCATEGORY_MAP:
        if any(word in cat for word in query.split()):
            cat_match = cat
            return cat_match, None
    return None, None

def extract_json_from_response(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract the largest JSON object
    matches = list(re.finditer(r'({.*?})', text, re.DOTALL))
    if matches:
        matches = sorted(matches, key=lambda m: len(m.group(0)), reverse=True)
        for match in matches:
            try:
                return json.loads(match.group(0))
            except Exception:
                continue
    # If still not valid, return None to trigger a retry
    return None

async def chat_with_agent(message: str, session_id: str, memory: dict):
    """
    Orchestrate chat with OpenAI GPT-4o, handle tool-calling, and manage memory.
    """
    # Extract category and subcategory
    category_slug, subcategory_slug = extract_category_and_subcategory(message)

    # Use memory if not found in current message
    if not category_slug:
        category_slug = memory.get("last_category")
    if not subcategory_slug:
        subcategory_slug = memory.get("last_subcategory")
    price_filter = extract_price_filter(message)
    def build_queryStr_from_price_filter(price_filter):
        if not price_filter:
            # If no filter, use full range
            return "price=[0,100000]"
        min_val = price_filter.get("$gte")
        max_val = price_filter.get("$lte")
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = min_val + 100000 if min_val is not None else 100000
        return f"price=[{min_val},{max_val}]"

    # Subcategory-aware logic
    if category_slug and price_filter:
        # If subcategory is present, check it first
        if subcategory_slug:
            subcat_check = await check_category(subcategory_slug, maincat=category_slug)
            if subcat_check and subcat_check.get('data', {}).get('exists', True):
                queryStr = build_queryStr_from_price_filter(price_filter)
                products = await get_products_by_category(category=category_slug, subcategory=subcategory_slug, queryStr=queryStr)
                # Fallback: if no products found, try without subcategory
                if not products:
                    products = await get_products_by_category(category=category_slug, subcategory="", queryStr=queryStr)
                    if products:
                        filtered = []
                        for prod in products:
                            price = parse_price(prod.get('product_mrp') or prod.get('price') or prod.get('product_price', ''))
                            if price is None:
                                continue
                            in_range = True
                            if "$lte" in price_filter and price > price_filter["$lte"]:
                                in_range = False
                            if "$gte" in price_filter and price < price_filter["$gte"]:
                                in_range = False
                            if in_range:
                                filtered.append(prod)
                        if not filtered:
                            return {
                                "status": "success",
                                "data": {
                                    "answer": f"Sorry, there are currently no {category_slug.replace('-', ' ')}s available in this price range. Would you like to see options in a higher price range?",
                                    "products": [],
                                    "end": ""
                                }
                            }
                        print(f"Products after filtering: {len(filtered)}")
                        for prod in filtered:
                            print(prod)
                        return {
                            "status": "success",
                            "data": {
                                "answer": f"Here are some {category_slug.replace('-', ' ')}s within your price range:",
                                "products": filtered,
                                "end": ""
                            }
                        }
                    return {
                        "status": "success",
                        "data": {
                            "answer": f"Sorry, there are currently no {subcategory_slug.replace('-', ' ')}s or {category_slug.replace('-', ' ')}s available in this price range. Would you like to see options in a higher price range?",
                            "products": [],
                            "end": ""
                        }
                    }
                if not products:
                    return {
                        "status": "success",
                        "data": {
                            "answer": f"Sorry, there are currently no {subcategory_slug.replace('-', ' ')}s available in this price range. Would you like to see options in a higher price range?",
                            "products": [],
                            "end": ""
                        }
                    }
                filtered = []
                for prod in products:
                    price = parse_price(prod.get('product_mrp') or prod.get('price') or prod.get('product_price', ''))
                    if price is None:
                        continue
                    in_range = True
                    if "$lte" in price_filter and price > price_filter["$lte"]:
                        in_range = False
                    if "$gte" in price_filter and price < price_filter["$gte"]:
                        in_range = False
                    if in_range:
                        filtered.append(prod)
                if not filtered:
                    return {
                        "status": "success",
                        "data": {
                            "answer": f"Sorry, there are currently no {subcategory_slug.replace('-', ' ')}s available in this price range. Would you like to see options in a higher price range?",
                            "products": [],
                            "end": ""
                        }
                    }
                print(f"Products after filtering: {len(filtered)}")
                for prod in filtered:
                    print(prod)
                return {
                    "status": "success",
                    "data": {
                        "answer": f"Here are some {subcategory_slug.replace('-', ' ')}s within your price range:",
                        "products": filtered,
                        "end": ""
                    }
                }
        # Fallback to just main category if no subcategory or subcategory doesn't exist
        cat_check = await check_category(category_slug)
        if not cat_check or not cat_check.get('data', {}).get('exists', True):
            return {
                "status": "success",
                "data": {
                    "answer": f"Sorry, the category '{category_slug}' does not exist.",
                    "products": [],
                    "end": ""
                }
            }
        queryStr = build_queryStr_from_price_filter(price_filter)
        products = await get_products_by_category(category=category_slug, queryStr=queryStr)
        if not products:
            return {
                "status": "success",
                "data": {
                    "answer": f"Sorry, there are currently no {category_slug.replace('-', ' ')}s available in this price range. Would you like to see options in a higher price range?",
                    "products": [],
                    "end": ""
                }
            }
        filtered = []
        for prod in products:
            price = parse_price(prod.get('product_mrp') or prod.get('price') or prod.get('product_price', ''))
            if price is None:
                continue
            in_range = True
            if "$lte" in price_filter and price > price_filter["$lte"]:
                in_range = False
            if "$gte" in price_filter and price < price_filter["$gte"]:
                in_range = False
            if in_range:
                filtered.append(prod)
        if not filtered:
            return {
                "status": "success",
                "data": {
                    "answer": f"Sorry, there are currently no {category_slug.replace('-', ' ')}s available in this price range. Would you like to see options in a higher price range?",
                    "products": [],
                    "end": ""
                }
            }
        print(f"Products after filtering: {len(filtered)}")
        for prod in filtered:
            print(prod)
        return {
            "status": "success",
            "data": {
                "answer": f"Here are some {category_slug.replace('-', ' ')}s within your price range:",
                "products": filtered,
                "end": ""
            }
        }
    # Prepare conversation history for OpenAI
    history = memory.get("history", [])
    # Always start with the system prompt
    # Only include the last 2 conversation turns (user and assistant) from history
    last_two_turns = []
    if history:
        # Find the last two user/assistant pairs
        # Each turn is a dict with 'role' and 'content'
        # We'll take up to the last 4 messages (user, assistant, user, assistant)
        last_two_turns = history[-4:]
    messages = (
        ([{"role": "system", "content": LOTUS_SYSTEM_PROMPT}]) +
        last_two_turns +
        [{"role": "user", "content": message}]
    )

    # Prepare function schemas for OpenAI
    function_schemas = [schema for _, schema in tool_registry.values()]

    # Call OpenAI with function-calling enabled
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=function_schemas,
        function_call="auto"
    )
    reply = response.choices[0].message

    # If the LLM wants to call a function/tool
    if reply.function_call:
        func_name = reply.function_call.name
        func_args = reply.function_call.arguments
        tool_func, _ = tool_registry.get(func_name, (None, None))
        if tool_func is None:
            tool_response = {"error": f"Tool '{func_name}' not found."}
        else:
            try:
                parsed_args = json.loads(func_args) if isinstance(func_args, str) else func_args

                # Only refine for get_products_by_category if category is missing
                if func_name == "get_products_by_category":
                    if not parsed_args.get("category") and memory.get("last_category"):
                        # Synthesize a query using last category/subcategory and current message
                        last_cat = memory["last_category"].replace("-", " ")
                        last_subcat = memory.get("last_subcategory", "")
                        if last_subcat:
                            last_subcat = last_subcat.replace("-", " ")
                            synthetic_query = f"{last_cat} {last_subcat} {message}"
                        else:
                            synthetic_query = f"{last_cat} {message}"
                        # Re-extract category/subcategory/price from the synthetic query
                        category_slug, subcategory_slug = extract_category_and_subcategory(synthetic_query)
                        price_filter = extract_price_filter(synthetic_query)
                        # Update parsed_args accordingly
                        parsed_args["category"] = category_slug
                        if subcategory_slug:
                            parsed_args["subcategory"] = subcategory_slug
                        # Optionally update queryStr if you use it

                # Inject auth_token from memory for get_orders if not present
                if func_name == "get_orders":
                    if "auth_token" not in parsed_args or not parsed_args["auth_token"]:
                        parsed_args["auth_token"] = memory.get("auth_token")
                # Await async tools
                if asyncio.iscoroutinefunction(tool_func):
                    tool_response = await tool_func(**parsed_args)
                else:
                    tool_response = tool_func(**parsed_args)
            except Exception as e:
                tool_response = {"error": str(e)}
        # Add tool call and result to messages
        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": func_name,
                "arguments": func_args
            }
        })
        messages.append({
            "role": "function",
            "name": func_name,
            "content": str(tool_response)
        })
        # Call LLM again to generate the final response
        response2 = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        final_reply = response2.choices[0].message.content
        # Extract JSON from the response
        final_json = extract_json_from_response(final_reply)
        if final_json is None:
            # Retry with a stricter prompt
            retry_prompt = (
                "Your previous response was not valid JSON. "
                "Please respond ONLY with a valid JSON object as instructed in the system prompt."
            )
            messages.append({"role": "user", "content": retry_prompt})
            response_retry = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            final_reply = response_retry.choices[0].message.content
            final_json = extract_json_from_response(final_reply)
            if final_json is None:
                # Still failed, return error
                return {
                    "status": "error",
                    "data": {
                        "answer": "Sorry, I could not process the response properly.",
                        "products": [],
                        "end": ""
                    }
                }
        # Update memory
        memory["history"] = messages + [{"role": "assistant", "content": final_reply}]
        if func_name in ["verify_otp", "sign_in"] and tool_response.get("error") == "0":
            auth_token = (
                tool_response.get("auth_token") or
                (tool_response.get("data", {}).get("auth_token") if isinstance(tool_response.get("data"), dict) else None)
            )
            if auth_token:
                memory["auth_token"] = auth_token
                memory["phone"] = parsed_args.get("phone")
                memory["is_authenticated"] = True
                if "data" in tool_response:
                    memory["user_data"] = tool_response["data"]
        # After a successful product search, update memory with last used category/subcategory
        if category_slug:
            memory["last_category"] = category_slug
        if subcategory_slug:
            memory["last_subcategory"] = subcategory_slug
        return final_json
    else:
        # Normal chat, no tool call - extract JSON from response
        final_json = extract_json_from_response(reply.content)
        if final_json is None:
            # Retry with a stricter prompt
            retry_prompt = (
                "Your previous response was not valid JSON. "
                "Please respond ONLY with a valid JSON object as instructed in the system prompt."
            )
            messages.append({"role": "user", "content": retry_prompt})
            response_retry = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            final_reply = response_retry.choices[0].message.content
            final_json = extract_json_from_response(final_reply)
            if final_json is None:
                # Still failed, return error
                final_json = {
                    "status": "error",
                    "data": {
                        "answer": "Sorry, I could not process the response properly.",
                        "products": [],
                        "end": ""
                    }
                }
        # Update memory
        memory["history"] = messages + [{"role": "assistant", "content": reply.content}]
        return final_json


def get_category_slug(user_query):
    query = user_query.lower().strip()
    for key, slug in CATEGORY_SUBCATEGORY_MAP.items():
        if key in query:
            return slug
    return None  # or a default/fallback

async def get_products_by_category_with_price_filter(user_query, subcategory=None, city=None, pincode=None):
    """
    Given a user query, extract category and price, check category, fetch products, and filter by price.
    Returns a dict with 'products' or an error message.
    """
    category_slug = get_category_slug(user_query)
    if not category_slug:
        return {"error": "Sorry, I couldn't find a matching category for your request."}

    # Check if category exists
    cat_check = await check_category(category_slug)
    if not cat_check or not cat_check.get('data', {}).get('exists', True):
        return {"error": f"Sorry, the category '{category_slug}' does not exist."}

    # Extract price filter
    price_filter = extract_price_filter(user_query)

    # Get products by category
    products = await get_products_by_category(
        category=category_slug,
        subcategory=subcategory,
        city=city,
        pincode=pincode
    )
    if not products:
        return {"error": "No products found in this category."}

    # Apply price filter if present
    if price_filter:
        filtered = []
        for prod in products:
            price = parse_price(prod.get('product_mrp') or prod.get('price') or prod.get('product_price', ''))
            if price is None:
                continue
            in_range = True
            if "$lte" in price_filter and price > price_filter["$lte"]:
                in_range = False
            if "$gte" in price_filter and price < price_filter["$gte"]:
                in_range = False
            if in_range:
                filtered.append(prod)
        products = filtered
        if not products:
            return {"error": "No products found in this category within the specified price range."}

    return {"products": products}


