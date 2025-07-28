import redis
import json

# Redis setup
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("Nawab",r.ping()) 
# Optional: Prefix and session expiry
SESSION_PREFIX = "chat_session:"
SESSION_TTL_SECONDS = 3600  # 1 hour

# Default memory template
def default_memory():
    return {
        "history": [],
        "is_authenticated": False,
        "auth_token": None,
        "last_category": None,
        "last_subcategory": None,
        "user_data": {},
        "phone": None,
    }

def add_chat_message(session_id: str, role: str, content: str):
    """
    Append a message to session memory.
    """
    memory = get_session_memory(session_id)
    history = memory.get("history", [])
    history.append({"role": role, "content": content})
    memory["history"] = history
    set_session_memory(session_id, memory)


def get_session_memory(session_id: str) -> dict:
    """
    Retrieve session memory for a given session ID.
    For authenticated users, fetch from Redis.
    For anonymous users, return default memory (and store it in Redis).
    """
    key = SESSION_PREFIX + session_id
    try:
        data = r.get(key)
        if data:
            return json.loads(data)
        memory = default_memory()
        r.setex(key, SESSION_TTL_SECONDS, json.dumps(memory))
        return memory
    except redis.exceptions.RedisError as e:
        print(f"[Redis Error - get_session_memory] {e}")
        return default_memory()

def set_session_memory(session_id: str, memory: dict):
    """
    Save session memory to Redis with expiry.
    """
    key = SESSION_PREFIX + session_id
    try:
        r.setex(key, SESSION_TTL_SECONDS, json.dumps(memory))
    except redis.exceptions.RedisError as e:
        print(f"[Redis Error - set_session_memory] {e}")

def clear_session_memory(session_id: str):
    """
    Delete session memory (e.g., on logout).
    """
    key = SESSION_PREFIX + session_id
    try:
        r.delete(key)
    except redis.exceptions.RedisError as e:
        print(f"[Redis Error - clear_session_memory] {e}")

def is_authenticated(session_id: str) -> bool:
    """
    Check if the session has an authenticated user.
    """
    memory = get_session_memory(session_id)
    return memory.get("is_authenticated", False)
