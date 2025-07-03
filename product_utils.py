import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ——— CONFIG ————————————————————————————————————————————————————————————
API_URL        = "https://portal.lotuselectronics.com/web-api/home/product_detail"
AUTH_HEADERS   = {
    "accept":             "application/json, text/plain, */*",
    "auth-key":           "Web2@!9",
    "auth-token":         "<YOUR_TOKEN_HERE>",
    "content-type":       "application/x-www-form-urlencoded",
    "end-client":         "Lotus-Web",
    "origin":             "https://www.lotuselectronics.com",
    "referer":            "https://www.lotuselectronics.com/",
    "user-agent":         "Mozilla/5.0 (Windows NT 10.0; Win64; x64)…"
}
REQUEST_TIMEOUT = 10
PRODUCT_ID_RE   = re.compile(r"/(\d+)(?:/|$)")

# ——— DATACLASS ————————————————————————————————————————————————————————
@dataclass
class RawDetail:
    instock:    str
    out_of_stock: str
    product_quantity: str
    product_image: Optional[list]
    product_images_350: Optional[list]

    @property
    def in_stock_flag(self) -> bool:
        return (
            self.instock.lower() == "yes"
            and self.out_of_stock == "0"
            and int(self.product_quantity or 0) > 0
        )

# ——— SESSION & RETRIES ——————————————————————————————————————————————
_session: Optional[requests.Session] = None

def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        _session = s
    return _session

# ——— HELPERS ———————————————————————————————————————————————————————
def extract_id(link: str) -> Optional[str]:
    # Try URL parse → regex fallback
    path = urlparse(link).path
    m = PRODUCT_ID_RE.search(path)
    return m.group(1) if m else None

def first_image(detail: RawDetail) -> Optional[str]:
    for attr in ("product_image", "product_images_350"):
        val = getattr(detail, attr)
        if isinstance(val, list) and val:
            return val[0]
    return None

# ——— CACHED RAW FETCH ——————————————————————————————————————————————
@lru_cache(maxsize=1024)
def _fetch_raw_detail(product_id: str) -> Optional[RawDetail]:
    payload = {
        "product_id":   product_id,
        "cat_name":     f"/product/{product_id}",
        "product_name": f"product-{product_id}",
    }
    try:
        resp = get_session().post(API_URL, headers=AUTH_HEADERS, data=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("product_detail")
        if not data:
            return None
        return RawDetail(**data)
    except Exception:
        return None

# ——— PUBLIC API —————————————————————————————————————————————————————
def get_product_stock_status(
    link: str
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns:
      in_stock (bool),
      first_image_url (or None),
      error_message (or None if OK)
    """
    pid = extract_id(link)
    if not pid:
        return False, None, "Invalid product URL"

    raw = _fetch_raw_detail(pid)
    if raw is None:
        return False, None, "No product details"

    return raw.in_stock_flag, first_image(raw), None

# ——— EXAMPLE —————————————————————————————————————————————————————————
if __name__ == "__main__":
    for url in [
        "https://www.lotuselectronics.com/product/full-hd-led-tv/tcl-full-hd-led-tv-80-cm-32-inches-32s5500af-black/38740",
        "https://www.lotuselectronics.com/product/invalid-url",
    ]:
        stock, img, err = get_product_stock_status(url)
        if err:
            print(f"[{url}] ERROR: {err}")
        else:
            print(f"[{url}] In stock={stock}, Image={img or 'n/a'}")
