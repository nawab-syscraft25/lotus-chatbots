from .search import search_products, search_products_schema
from .vector import vector_search_products, vector_search_schema
from .auth import check_user, check_user_schema, send_otp, send_otp_schema, verify_otp, verify_otp_schema
from .order import get_orders, get_orders_schema

tool_registry = {
    "search_products": (search_products, search_products_schema),
    "vector_search_products": (vector_search_products, vector_search_schema),
    "check_user": (check_user, check_user_schema),
    "send_otp": (send_otp, send_otp_schema),
    "verify_otp": (verify_otp, verify_otp_schema),
    "get_orders": (get_orders, get_orders_schema),
}

def is_authenticated(memory: dict) -> bool:
    return bool(memory.get("auth_token")) 

