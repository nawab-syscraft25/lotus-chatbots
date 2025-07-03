from .search import search_products, search_products_schema
from .vector import vector_search_products, vector_search_schema
from .auth import check_user, check_user_schema, send_otp, send_otp_schema, verify_otp, verify_otp_schema
from .order import get_orders, get_orders_schema
from .offers import get_current_offers, get_current_offers_schema
from .check_delivery import check_product_delivery, check_product_delivery_schema
from .near_stores import check_near_stores, check_near_stores_schema
from .get_products_by_category import get_products_by_category, get_products_by_category_schema
from .check_category import check_category, check_category_schema

tool_registry = {
    "search_products": (search_products, search_products_schema),
    "vector_search_products": (vector_search_products, vector_search_schema),
    "check_user": (check_user, check_user_schema),
    "send_otp": (send_otp, send_otp_schema),
    "verify_otp": (verify_otp, verify_otp_schema),
    "get_orders": (get_orders, get_orders_schema),
    "get_current_offers": (get_current_offers, get_current_offers_schema),
    "check_product_delivery": (check_product_delivery, check_product_delivery_schema),
    "check_near_stores": (check_near_stores, check_near_stores_schema),
    "get_products_by_category": (get_products_by_category, get_products_by_category_schema),
    # "check_category": (check_category, check_category_schema),
}

def is_authenticated(memory: dict) -> bool:
    return bool(memory.get("auth_token")) 

