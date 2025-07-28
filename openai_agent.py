# openai_agent.py

import os
import openai
import asyncio
import json
import re
import traceback
import sys
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append('.')

# Import dependencies
try:
    from tools.tool_registry import tool_registry
    from vector_search import extract_price_filter, parse_price
    from memory.memory_store import get_session_memory, set_session_memory, is_authenticated
    import google.generativeai as genai
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# Configure APIs
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ResponseStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

@dataclass
class ChatResponse:
    status: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "data": self.data
        }

class LotusSystemPrompt:
    """Centralized system prompt management"""
    
    BASE_PROMPT = """You are Lotus, the official AI assistant for Lotus Electronics.
Your role: provide accurate product details, store information, order status, and shopping guidance — always as a friendly Lotus representative.

CORE RESPONSIBILITIES:
- Product search, recommendations, and comparisons
- Order status and user account management
- Store information and delivery details
- Shopping assistance and guidance

TOOL DECISION RULES:
1. Product Queries:
   • General searches ('latest iPhone price', 'Samsung TVs'): use search_products
   • Category + budget ('laptops under 50k', 'headphones above 5k'): use get_products_by_category
   • Contextual/preference searches ('gaming headphones', 'travel laptop'): use vector_search_products
   • Delivery checks: use check_product_delivery

2. User Management:
   • Order queries (authenticated users): use get_orders with auth_token
   • Authentication: use send_otp → verify_otp flow
   • Never reveal private user data

3. Comparisons:
   • When requested, include 'comparison' field in JSON response
   • Format: [{"name": "...", "vs_name": "...", "differences": [...]}]

RESPONSE FORMAT (STRICT JSON ONLY):
{
  "status": "success|error|partial",
  "data": {
    "answer": "Your helpful response here",
    "products": [{"name": "...", "link": "...", "price": "₹...", "image": "...", "features": [...]}],
    "comparison": [{"name": "...", "vs_name": "...", "differences": [...]}], // optional
    "end": "Follow-up question or next steps"
  }
}

GUIDELINES:
- Always use tools for product data - never invent product information
- For unclear queries, ask clarifying questions
- Keep responses conversational and helpful
- Focus on features that matter to customers
- If no products found, suggest alternatives
- Include follow-up questions to help users decide
"""

    @classmethod
    def get_prompt(cls) -> str:
        return cls.BASE_PROMPT

class CategoryMatcher:
    """Enhanced category and subcategory matching"""
    
    CATEGORY_SUBCATEGORY_MAP = {
        "led-television": [
            "smart-tv", "qled-tv", "oled-tv", "4k-ultra-hd-tv", "8k-ultra-hd-tv", 
            "full-hd-led-tv", "hd-led-tv"
        ],
        "mobile-phone-tablet": [
            "mobile-phones", "smart-wearables-fitness-band", "tablets-ipads", 
            "corded-telephone", "mobile-accessories"
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
            "convertible-laptop", "windows-laptop", "gaming-laptop", 
            "macbook-laptop", "thin-light-laptop"
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
            "audio-accessories", "gaming", "gaming-software", "headphones", 
            "home-theatre", "projectors", "sound-bar", "speakers"
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
            "air-conditioners", "deep-freezer", "dishwasher", "electronic-safe", 
            "fans", "floor-home-care", "irons-garment-steamer", "refrigerators", 
            "room-coolers", "room-heaters", "sewing-machines", "washing-machines", 
            "water-heaters", "water-purifiers-dispensers"
        ],
        "air-conditioners": [
            "cassette-ac", "portable-ac", "tower-ac", "wall-mounted-split-ac", "window-ac"
        ],
        "fans": [
            "ceiling-fan", "exhaust-fan", "pedestal-fan", "table-fan", "tower-fan", "wall-fan"
        ],
        "refrigerators": [
            "bottom-freezer-refrigerator", "built-in-refrigerator", "double-door-refrigerator", 
            "french-door-refrigerator", "mini-refrigerator", "side-by-side-refrigerator", 
            "single-door-refrigerator", "triple-door-refrigerator"
        ],
        "washing-machines": [
            "cloth-dryer", "front-load-washing-machine", "semi-automatic-washing-machine", 
            "top-load-washing-machine"
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
            "attamakers", "beverage-makers", "chimneys", "cooking-appliances", 
            "food-processor", "juicer-mixer-grinder", "microwave-ovens", 
            "sandwich-maker", "toasters-grillers"
        ],
        "beverage-makers": [
            "electric-kettle", "tea-coffee-maker"
        ],
        "cooking-appliances": [
            "air-fryer", "built-in-hob", "built-in-oven", "cooking-hob", "gas-stove", 
            "induction-cooker", "otg", "pop-up-toaster", "rice-induction-cookers"
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
            "hair-brush", "hair-curler", "hair-dryer", "hair-multi-stiller", 
            "hair-straightener", "hair-styler"
        ],
        "hair-grooming": [
            "beard-trimmer", "body-groomer", "hair-brush", "hair-curler", 
            "hair-dryer", "hair-grooming-kit", "hair-multi-stiller", 
            "hair-straightener", "hair-styler", "multi-groomer"
        ],
        "corded-telephone": [
            "cordless-phones", "landline-phones"
        ]
    }
    
    # Common synonyms and variations
    SYNONYMS = {
        "mobile": ["phone", "smartphone", "cell phone"],
        "laptop": ["notebook", "computer"],
        "tv": ["television", "smart tv"],
        "ac": ["air conditioner", "cooling"],
        "fridge": ["refrigerator", "cooling"],
        "headphone": ["earphone", "headset"],
        "mixer": ["grinder", "juicer"]
    }
    
    @classmethod
    def extract_category_and_subcategory(cls, user_query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract category and subcategory from user query with fuzzy matching"""
        query = user_query.lower().replace('&', 'and').replace('  ', ' ').strip()
        
        # Direct subcategory match (most specific)
        for category, subcategories in cls.CATEGORY_SUBCATEGORY_MAP.items():
            for subcategory in subcategories:
                # Exact match
                if subcategory.replace('-', ' ') in query or subcategory in query:
                    return category, subcategory
                
                # Fuzzy match with synonyms
                subcat_words = subcategory.replace('-', ' ').split()
                if any(word in query for word in subcat_words):
                    return category, subcategory
        
        # Category match
        for category in cls.CATEGORY_SUBCATEGORY_MAP:
            if category.replace('-', ' ') in query or category in query:
                return category, None
            
            # Fuzzy match for categories
            cat_words = category.replace('-', ' ').split()
            if any(word in query for word in cat_words):
                return category, None
        
        # Synonym-based matching
        for term, synonyms in cls.SYNONYMS.items():
            if any(syn in query for syn in synonyms + [term]):
                for category, subcategories in cls.CATEGORY_SUBCATEGORY_MAP.items():
                    for subcategory in subcategories:
                        if term in subcategory.replace('-', ' '):
                            return category, subcategory
        
        return None, None
    
    @classmethod
    def build_llm_prompt(cls, user_query: str) -> str:
        """Build prompt for LLM-based category extraction"""
        flat_list = []
        for cat, subcats in cls.CATEGORY_SUBCATEGORY_MAP.items():
            flat_list.append(f"- {cat}:\n  " + ", ".join(subcats))
        cat_map_str = "\n".join(flat_list)

        return f"""
You are an intelligent assistant helping to classify user queries into categories and subcategories for an electronics store.

### Product Categories and Subcategories:
{cat_map_str}

### Task:
Analyze the user query and extract the **most appropriate category and subcategory**.

### Output Format:
Respond ONLY with this JSON format:
{{"category": "category-slug", "subcategory": "subcategory-slug"}}

If only category matches, set subcategory to null.
If no match found, set both to null.

### User Query:
{user_query.strip()}
"""

    @classmethod
    def get_category_subcategory_llm(cls, user_query: str) -> Tuple[Optional[str], Optional[str]]:
        """Use LLM for category extraction with fallback to rule-based"""
        try:
            # First try rule-based extraction
            category, subcategory = cls.extract_category_and_subcategory(user_query)
            if category:
                return category, subcategory
            
            # Fallback to LLM
            prompt = cls.build_llm_prompt(user_query)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            
            result = cls._extract_json_from_text(response.text)
            return result.get("category"), result.get("subcategory")
            
        except Exception as e:
            logger.error(f"LLM category extraction error: {e}")
            return None, None
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
        except Exception:
            pass
        return {"category": None, "subcategory": None}

class JSONResponseHandler:
    """Handle JSON response extraction and validation"""
    
    @staticmethod
    def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
        """Enhanced JSON extraction with multiple fallback strategies"""
        if not response_text:
            return None
        
        # Strategy 1: Parse entire response as JSON
        try:
            parsed = json.loads(response_text.strip())
            if JSONResponseHandler._is_valid_chat_response(parsed):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find JSON in code blocks or patterns
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*"status"[^{}]*\})',
            r'(\{.*?"data".*?\})',
            r'(\{.*?\})'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if JSONResponseHandler._is_valid_chat_response(parsed):
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Create default response from text
        return JSONResponseHandler._create_default_response(response_text)
    
    @staticmethod
    def _is_valid_chat_response(data: Any) -> bool:
        """Validate if response has expected structure"""
        return (
            isinstance(data, dict) and 
            "status" in data and 
            "data" in data and 
            isinstance(data["data"], dict)
        )
    
    @staticmethod
    def _create_default_response(text: str) -> Dict[str, Any]:
        """Create default response structure"""
        return {
            "status": ResponseStatus.SUCCESS.value,
            "data": {
                "answer": text.strip() if text else "I understand your request.",
                "products": [],
                "end": ""
            }
        }
    
    @staticmethod
    def create_error_response(message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "status": ResponseStatus.ERROR.value,
            "data": {
                "answer": message,
                "products": [],
                "end": ""
            }
        }

class ProductFilter:
    """Handle product filtering by price and other criteria"""
    
    @staticmethod
    def filter_by_price(products: List[Dict], price_filter: Dict[str, Any]) -> List[Dict]:
        """Filter products by price range"""
        if not price_filter or not products:
            return products
        
        filtered = []
        for product in products:
            try:
                price = parse_price(
                    product.get('product_mrp') or 
                    product.get('price') or 
                    product.get('product_price', '')
                )
                
                if price is None:
                    continue
                
                # Check price range
                if "$gte" in price_filter and price < price_filter["$gte"]:
                    continue
                if "$lte" in price_filter and price > price_filter["$lte"]:
                    continue
                
                filtered.append(product)
                
            except Exception as e:
                logger.warning(f"Error parsing price for product: {e}")
                continue
        
        return filtered
    
    @staticmethod
    def build_price_query_string(price_filter: Dict[str, Any]) -> str:
        """Build query string from price filter"""
        if not price_filter:
            return "price=[0,500000]"
        
        try:
            min_val = max(0, price_filter.get("$gte", 0))
            max_val = price_filter.get("$lte", 500000)
            
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            
            return f"price=[{int(min_val)},{int(max_val)}]"
            
        except Exception as e:
            logger.error(f"Error building price query: {e}")
            return "price=[0,500000]"

class MemoryManager:
    """Enhanced memory management for sessions"""
    
    @staticmethod
    def get_safe_memory(session_id: str) -> Dict[str, Any]:
        """Get session memory with error handling"""
        try:
            memory = get_session_memory(session_id)
            return memory if isinstance(memory, dict) else {}
        except Exception as e:
            logger.error(f"Error getting session memory: {e}")
            return {}
    
    @staticmethod
    def save_safe_memory(session_id: str, memory: Dict[str, Any]) -> bool:
        """Save session memory with error handling"""
        try:
            set_session_memory(session_id, memory)
            return True
        except Exception as e:
            logger.error(f"Error saving session memory: {e}")
            return False
    
    @staticmethod
    def update_conversation_history(memory: Dict[str, Any], messages: List[Dict]) -> None:
        """Update conversation history in memory"""
        try:
            # Filter out function calls for cleaner history
            clean_messages = [
                msg for msg in messages 
                if msg.get("role") != "function" and msg.get("content") is not None
            ]
            
            # Keep last 10 messages for context
            memory["history"] = clean_messages[-10:]
            
        except Exception as e:
            logger.error(f"Error updating conversation history: {e}")
    
    @staticmethod
    def handle_authentication_success(memory: Dict[str, Any], tool_response: Dict, 
                                    parsed_args: Dict) -> None:
        """Handle successful authentication"""
        try:
            if isinstance(tool_response, dict):
                # Check for success indicators
                success_indicators = [
                    tool_response.get("error") == "0",
                    tool_response.get("status") == "success",
                    tool_response.get("success") is True
                ]
                
                if any(success_indicators):
                    # Extract auth token
                    auth_token = (
                        tool_response.get("auth_token") or
                        (tool_response.get("data", {}).get("auth_token") 
                         if isinstance(tool_response.get("data"), dict) else None)
                    )
                    
                    if auth_token:
                        memory["auth_token"] = auth_token
                        memory["phone"] = parsed_args.get("phone")
                        memory["is_authenticated"] = True
                        
                        if "data" in tool_response:
                            memory["user_data"] = tool_response["data"]
                            
        except Exception as e:
            logger.error(f"Error handling authentication success: {e}")

class LotusAgent:
    """Main agent class for handling chat interactions"""
    
    def __init__(self):
        self.category_matcher = CategoryMatcher()
        self.json_handler = JSONResponseHandler()
        self.product_filter = ProductFilter()
        self.memory_manager = MemoryManager()
        
    async def chat_with_agent(self, message: str, session_id: str) -> Dict[str, Any]:
        """Main chat handler with comprehensive error handling"""
        try:
            # Input validation
            if not message or not message.strip():
                return self.json_handler.create_error_response(
                    "Please provide a message."
                )
            
            if not session_id:
                return self.json_handler.create_error_response(
                    "Session ID is required."
                )
            
            # Get session memory
            memory = self.memory_manager.get_safe_memory(session_id)
            
            # Extract category and price information
            category_slug, subcategory_slug = await self._extract_category_info(
                message, memory
            )
            price_filter = await self._extract_price_info(message)
            
            # Check for direct product search opportunity
            if category_slug and price_filter:
                response = await self._handle_direct_product_search(
                    category_slug, subcategory_slug, price_filter, memory, session_id
                )
                if response:
                    return response
            
            # Proceed with OpenAI conversation
            return await self._handle_openai_conversation(
                message, session_id, memory, category_slug, subcategory_slug
            )
            
        except Exception as e:
            logger.error(f"Error in chat_with_agent: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return self.json_handler.create_error_response(
                "I encountered an error while processing your request. Please try again."
            )
    
    async def _extract_category_info(self, message: str, memory: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract category and subcategory information"""
        try:
            category_slug, subcategory_slug = self.category_matcher.get_category_subcategory_llm(message)
            
            # Use memory fallback if not found
            if not category_slug:
                category_slug = memory.get("last_category")
            if not subcategory_slug:
                subcategory_slug = memory.get("last_subcategory")
            
            return category_slug, subcategory_slug
            
        except Exception as e:
            logger.error(f"Error extracting category info: {e}")
            return None, None
    
    async def _extract_price_info(self, message: str) -> Optional[Dict]:
        """Extract price filter information"""
        try:
            return extract_price_filter(message)
        except Exception as e:
            logger.error(f"Error extracting price info: {e}")
            return None
    
    async def _handle_direct_product_search(self, category_slug: str, subcategory_slug: str,
                                          price_filter: Dict, memory: Dict, 
                                          session_id: str) -> Optional[Dict]:
        """Handle direct product search when category and price are available"""
        try:
            from tools.get_products_by_category import get_products_by_category
            
            # Build query string
            query_str = self.product_filter.build_price_query_string(price_filter)
            
            # Search with subcategory first
            products = await get_products_by_category(
                category=category_slug,
                subcategory=subcategory_slug or "",
                queryStr=query_str
            )
            
            # Fallback to category only if no products found
            if not products and subcategory_slug:
                products = await get_products_by_category(
                    category=category_slug,
                    subcategory="",
                    queryStr=query_str
                )
            
            # Filter by price
            if products:
                filtered_products = self.product_filter.filter_by_price(products, price_filter)
                return self._build_product_response(
                    filtered_products, category_slug, subcategory_slug, memory, session_id
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in direct product search: {e}")
            return None
    
    def _build_product_response(self, products: List[Dict], category_slug: str,
                              subcategory_slug: str, memory: Dict, session_id: str) -> Dict:
        """Build standardized product response"""
        try:
            # Update memory
            memory["last_category"] = category_slug
            if subcategory_slug:
                memory["last_subcategory"] = subcategory_slug
            self.memory_manager.save_safe_memory(session_id, memory)
            
            # Build response message
            if not products:
                if subcategory_slug:
                    answer = f"Sorry, no {subcategory_slug.replace('-', ' ')}s found in your price range. Would you like to see options in a different price range?"
                else:
                    answer = f"Sorry, no {category_slug.replace('-', ' ')}s found in your price range. Would you like to see options in a different price range?"
            else:
                if subcategory_slug:
                    answer = f"Here are some {subcategory_slug.replace('-', ' ')}s within your price range:"
                else:
                    answer = f"Here are some {category_slug.replace('-', ' ')}s within your price range:"
            
            return {
                "status": ResponseStatus.SUCCESS.value,
                "data": {
                    "answer": answer,
                    "products": products[:10],  # Limit to 10 products
                    "end": "Would you like to see more details about any of these products?"
                }
            }
            
        except Exception as e:
            logger.error(f"Error building product response: {e}")
            return self.json_handler.create_error_response(
                "Error processing product search results."
            )
    
    async def _handle_openai_conversation(self, message: str, session_id: str, memory: Dict,
                                        category_slug: str, subcategory_slug: str) -> Dict:
        """Handle OpenAI conversation with function calling"""
        try:
            # Prepare conversation history
            history = memory.get("history", [])
            if not isinstance(history, list):
                history = []
            
            # Build messages for OpenAI
            messages = self._build_openai_messages(history, message)
            
            # Get function schemas
            function_schemas = [schema for _, schema in tool_registry.values()]
            
            # Call OpenAI
            response = await self._call_openai(messages, function_schemas)
            
            if not response or not response.choices:
                raise Exception("Empty response from OpenAI")
            
            reply = response.choices[0].message
            
            # Handle function calls or normal response
            if hasattr(reply, 'function_call') and reply.function_call:
                return await self._handle_function_call(
                    reply, messages, memory, session_id, category_slug, subcategory_slug
                )
            else:
                return await self._handle_normal_response(
                    reply, messages, memory, session_id
                )
                
        except Exception as e:
            logger.error(f"Error in OpenAI conversation: {e}")
            return self.json_handler.create_error_response(
                "I encountered an error while processing your request. Please try again."
            )
    
    def _build_openai_messages(self, history: List[Dict], message: str) -> List[Dict]:
        """Build messages for OpenAI API"""
        # System message
        messages = [{"role": "system", "content": LotusSystemPrompt.get_prompt()}]
        
        # Add conversation history (last 8 messages)
        for msg in history[-8:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["content"] is None:
                    msg["content"] = ""
                messages.append(msg)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    async def _call_openai(self, messages: List[Dict], function_schemas: List[Dict]) -> Any:
        """Call OpenAI API with proper configuration"""
        return openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=function_schemas if function_schemas else None,
            function_call="auto" if function_schemas else None,
            temperature=0.7,
            max_tokens=2000,
            timeout=30  # Add timeout
        )
    
    async def _handle_function_call(self, reply: Any, messages: List[Dict], memory: Dict,
                                  session_id: str, category_slug: str, 
                                  subcategory_slug: str) -> Dict:
        """Handle OpenAI function calls"""
        try:
            func_name = reply.function_call.name
            func_args = reply.function_call.arguments
            
            # Get tool function
            tool_func, _ = tool_registry.get(func_name, (None, None))
            
            if tool_func is None:
                tool_response = {"error": f"Tool '{func_name}' not found."}
            else:
                # Parse arguments
                try:
                    parsed_args = json.loads(func_args) if isinstance(func_args, str) else func_args
                    if not isinstance(parsed_args, dict):
                        parsed_args = {}
                except json.JSONDecodeError:
                    parsed_args = {}
                
                # Handle specific tool configurations
                parsed_args = self._configure_tool_args(
                    func_name, parsed_args, memory, messages, category_slug, subcategory_slug
                )
                
                # Execute tool
                try:
                    if asyncio.iscoroutinefunction(tool_func):
                        tool_response = await tool_func(**parsed_args)
                    else:
                        tool_response = tool_func(**parsed_args)
                except Exception as e:
                    logger.error(f"Error executing tool {func_name}: {e}")
                    tool_response = {"error": f"Tool execution failed: {str(e)}"}
            
            # Add function call to message history
            messages.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": func_name,
                        "arguments": func_args
                    }
                },
                {
                    "role": "function",
                    "name": func_name,
                    "content": str(tool_response)
                }
            ])
            
            # Get final response from OpenAI
            final_response = await self._get_final_openai_response(messages)
            
            # Handle authentication success
            if func_name in ["verify_otp", "sign_in"]:
                self.memory_manager.handle_authentication_success(
                    memory, tool_response, parsed_args
                )
            
            # Update memory
            self.memory_manager.update_conversation_history(memory, messages)
            self.memory_manager.save_safe_memory(session_id, memory)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error handling function call: {e}")
            return self.json_handler.create_error_response(
                "Error processing your request. Please try again."
            )
    
    def _configure_tool_args(self, func_name: str, parsed_args: Dict, memory: Dict,
                           messages: List[Dict], category_slug: str, 
                           subcategory_slug: str) -> Dict:
        """Configure tool arguments based on function type and context"""
        try:
            # Handle category-based product search
            if func_name == "get_products_by_category":
                if not parsed_args.get("category") and memory.get("last_category"):
                    # Use memory category if not provided
                    parsed_args["category"] = memory["last_category"]
                    
                    if not parsed_args.get("subcategory") and memory.get("last_subcategory"):
                        parsed_args["subcategory"] = memory["last_subcategory"]
                
                # Ensure category is provided
                if not parsed_args.get("category") and category_slug:
                    parsed_args["category"] = category_slug
                if not parsed_args.get("subcategory") and subcategory_slug:
                    parsed_args["subcategory"] = subcategory_slug
            
            # Handle authentication-required tools
            elif func_name in ["get_orders", "get_user_profile"]:
                if "auth_token" not in parsed_args or not parsed_args["auth_token"]:
                    parsed_args["auth_token"] = memory.get("auth_token")
            
            # Handle delivery checks
            elif func_name == "check_product_delivery":
                if not parsed_args.get("pincode") and memory.get("user_pincode"):
                    parsed_args["pincode"] = memory["user_pincode"]
            
            return parsed_args
            
        except Exception as e:
            logger.error(f"Error configuring tool args: {e}")
            return parsed_args
    
    async def _get_final_openai_response(self, messages: List[Dict]) -> Dict:
        """Get final response from OpenAI after function call"""
        try:
            response = await self._call_openai(messages, [])
            
            if not response or not response.choices:
                raise Exception("Empty response from OpenAI")
            
            final_reply = response.choices[0].message.content
            
            # Extract and validate JSON response
            final_json = self.json_handler.extract_json_from_response(final_reply)
            
            if final_json is None:
                # Retry with stricter prompt
                retry_prompt = (
                    "Your previous response was not valid JSON. "
                    "Please respond ONLY with valid JSON in this format: "
                    '{"status": "success", "data": {"answer": "your answer", "products": [], "end": ""}}'
                )
                
                messages.append({"role": "user", "content": retry_prompt})
                
                retry_response = await self._call_openai(messages, [])
                if retry_response and retry_response.choices:
                    final_reply = retry_response.choices[0].message.content
                    final_json = self.json_handler.extract_json_from_response(final_reply)
            
            # Final fallback
            if final_json is None:
                final_json = self.json_handler.create_error_response(
                    "I processed your request but had trouble formatting the response. Please try again."
                )
            
            return final_json
            
        except Exception as e:
            logger.error(f"Error getting final OpenAI response: {e}")
            return self.json_handler.create_error_response(
                "Error generating final response."
            )
    
    async def _handle_normal_response(self, reply: Any, messages: List[Dict], 
                                    memory: Dict, session_id: str) -> Dict:
        """Handle normal chat response without function calls"""
        try:
            # Extract JSON from response
            final_json = self.json_handler.extract_json_from_response(reply.content)
            
            # Validate response structure
            if not self.json_handler._is_valid_chat_response(final_json):
                final_json = self.json_handler._create_default_response(
                    reply.content if reply.content else "I understand your request."
                )
            
            # Check for potential hallucinated products
            products = final_json.get("data", {}).get("products", [])
            if products and len(products) > 0:
                # Verify products have required fields
                valid_products = [
                    p for p in products 
                    if isinstance(p, dict) and p.get("name") and p.get("price")
                ]
                
                if not valid_products:
                    # Remove hallucinated products
                    final_json["data"]["products"] = []
                    final_json["data"]["answer"] = (
                        "I'd be happy to help you find products. "
                        "Could you please specify what type of product you're looking for?"
                    )
            
            # Update memory
            self.memory_manager.update_conversation_history(
                memory, messages + [{"role": "assistant", "content": reply.content}]
            )
            self.memory_manager.save_safe_memory(session_id, memory)
            
            return final_json
            
        except Exception as e:
            logger.error(f"Error handling normal response: {e}")
            return self.json_handler.create_error_response(
                "Error processing response."
            )

# Utility functions for backward compatibility and additional features
class UtilityFunctions:
    """Additional utility functions for the chatbot"""
    
    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """Validate Indian phone number format"""
        if not phone:
            return False
        
        # Remove spaces, dashes, and country code
        clean_phone = re.sub(r'[\s\-\+]', '', phone)
        if clean_phone.startswith('91'):
            clean_phone = clean_phone[2:]
        
        # Check if it's a valid 10-digit number
        return len(clean_phone) == 10 and clean_phone.isdigit()
    
    @staticmethod
    def format_price(price: str) -> str:
        """Format price consistently"""
        try:
            if not price:
                return "Price not available"
            
            # Remove currency symbols and clean
            clean_price = re.sub(r'[^\d.]', '', str(price))
            if not clean_price:
                return "Price not available"
            
            price_float = float(clean_price)
            
            # Format with Indian number system
            if price_float >= 10000000:  # 1 crore
                return f"₹{price_float/10000000:.1f} Cr"
            elif price_float >= 100000:  # 1 lakh
                return f"₹{price_float/100000:.1f} L"
            elif price_float >= 1000:  # 1 thousand
                return f"₹{price_float/1000:.1f}K"
            else:
                return f"₹{int(price_float)}"
                
        except Exception:
            return "Price not available"
    
    @staticmethod
    def extract_features_from_product(product: Dict) -> List[str]:
        """Extract and format product features"""
        features = []
        
        try:
            # Common feature fields
            feature_fields = [
                'features', 'specifications', 'key_features', 
                'highlights', 'product_features'
            ]
            
            for field in feature_fields:
                if field in product and product[field]:
                    if isinstance(product[field], list):
                        features.extend(product[field])
                    elif isinstance(product[field], str):
                        # Split by common delimiters
                        parts = re.split(r'[,;|]', product[field])
                        features.extend([p.strip() for p in parts if p.strip()])
            
            # Extract from description if no features found
            if not features and product.get('description'):
                desc = product['description']
                # Look for feature patterns
                feature_patterns = [
                    r'(\d+(?:\.\d+)?\s*(?:inch|"|GB|MP|Hz|W|V|A))',  # Technical specs
                    r'(wireless|bluetooth|smart|led|4k|hd|android|ios)',  # Keywords
                ]
                
                for pattern in feature_patterns:
                    matches = re.findall(pattern, desc, re.IGNORECASE)
                    features.extend(matches)
            
            # Clean and limit features
            clean_features = []
            for feature in features[:6]:  # Limit to 6 features
                if isinstance(feature, str) and len(feature.strip()) > 0:
                    clean_features.append(feature.strip()[:100])  # Limit length
            
            return clean_features if clean_features else ["Product specifications available"]
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return ["Product specifications available"]

# Enhanced error handling and monitoring
class ErrorHandler:
    """Centralized error handling and monitoring"""
    
    @staticmethod
    def log_error(error: Exception, context: str, **kwargs):
        """Log error with context"""
        logger.error(f"Error in {context}: {str(error)}")
        logger.error(f"Error type: {type(error).__name__}")
        logger.error(f"Context data: {kwargs}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    @staticmethod
    def create_fallback_response(error_type: str = "general") -> Dict[str, Any]:
        """Create appropriate fallback response based on error type"""
        fallback_messages = {
            "general": "I'm experiencing some technical difficulties. Please try again in a moment.",
            "api": "I'm having trouble connecting to our services. Please try again.",
            "parsing": "I had trouble understanding that. Could you please rephrase your request?",
            "authentication": "There was an issue with authentication. Please try logging in again.",
            "product": "I couldn't retrieve product information right now. Please try again."
        }
        
        message = fallback_messages.get(error_type, fallback_messages["general"])
        
        return {
            "status": ResponseStatus.ERROR.value,
            "data": {
                "answer": message,
                "products": [],
                "end": ""
            }
        }

# Main chat function for backward compatibility
async def chat_with_agent(message: str, session_id: str) -> Dict[str, Any]:
    """
    Main entry point for chat interactions
    Enhanced with comprehensive error handling and robust architecture
    """
    agent = LotusAgent()
    return await agent.chat_with_agent(message, session_id)

# Additional helper functions
def get_category_subcategory_llm(user_query: str) -> Tuple[Optional[str], Optional[str]]:
    """Helper function for category extraction"""
    return CategoryMatcher.get_category_subcategory_llm(user_query)

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Helper function for JSON extraction"""
    return JSONResponseHandler.extract_json_from_response(response_text)

# Configuration and initialization
def initialize_agent():
    """Initialize the agent with required configurations"""
    try:
        # Verify required environment variables
        required_env_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            raise Exception(f"Missing required environment variables: {missing_vars}")
        
        # Verify tool registry
        if not tool_registry:
            logger.warning("Tool registry is empty")
        
        logger.info("Lotus Agent initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return False

# Initialize on import
if __name__ != "__main__":
    initialize_agent()