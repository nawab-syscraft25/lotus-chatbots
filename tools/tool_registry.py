from .search import search_products, search_products_schema
from .vector import vector_search_products, vector_search_schema
from .check_delivery import check_product_delivery, check_product_delivery_schema
from .get_products_by_category import get_products_by_category, get_products_by_category_schema


tool_registry = {
    "search_products": (search_products, search_products_schema),
    "vector_search_products": (vector_search_products, vector_search_schema),
    "check_product_delivery": (check_product_delivery, check_product_delivery_schema),
    "get_products_by_category": (get_products_by_category, get_products_by_category_schema),
}


