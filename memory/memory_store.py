# memory/memory_store.py

from .database import db_manager
from typing import Dict, List

# In-memory storage for anonymous users (fallback)
_session_memory = {}

def get_session_memory(session_id: str) -> Dict:
    """
    Get session memory from database if authenticated, otherwise from in-memory storage.
    Only authenticated users get persistent storage.
    """
    # First check if session exists in database
    session_data = db_manager.get_session_data(session_id)
    
    if session_data and session_data.get("auth_token"):
        # User is authenticated - use database storage
        history = db_manager.get_chat_history(session_id)
        return {
            "history": history,
            "auth_token": session_data["auth_token"],
            "phone": session_data["phone"],
            "user_data": session_data["user_data"],
            "user_id": session_data["user_id"],
            "is_authenticated": True
        }
    else:
        # Anonymous user - use in-memory storage
        if session_id not in _session_memory:
            _session_memory[session_id] = {"history": [], "is_authenticated": False}
        return _session_memory[session_id]

def update_session_memory(session_id: str, memory: Dict) -> bool:
    """
    Update session memory. For authenticated users, save to database.
    For anonymous users, update in-memory storage.
    """
    if memory.get("is_authenticated"):
        # Update database for authenticated users
        db_manager.update_session_activity(session_id)
        return True
    else:
        # Update in-memory storage for anonymous users
        _session_memory[session_id] = memory
        return True

def add_chat_message(session_id: str, role: str, content: str) -> bool:
    """
    Add a chat message to history. For authenticated users, save to database.
    For anonymous users, add to in-memory storage.
    """
    session_data = db_manager.get_session_data(session_id)
    
    if session_data and session_data.get("auth_token"):
        # Save to database for authenticated users
        return db_manager.add_chat_message(session_id, role, content)
    else:
        # Add to in-memory storage for anonymous users
        if session_id not in _session_memory:
            _session_memory[session_id] = {"history": [], "is_authenticated": False}
        
        _session_memory[session_id]["history"].append({
            "role": role,
            "content": content,
            "timestamp": None  # In-memory doesn't track timestamps
        })
        return True

def authenticate_user(session_id: str, phone: str, auth_token: str, user_data: Dict = None) -> bool:
    """
    Authenticate a user and migrate their data to database storage.
    """
    try:
        # Create or update user in database
        user_id = db_manager.create_or_update_user(phone, auth_token, user_data)
        
        # Create or update session with authentication data
        success = db_manager.update_session_auth(session_id, user_id, auth_token, phone)
        
        if success:
            # Migrate any existing in-memory history to database
            if session_id in _session_memory:
                history = _session_memory[session_id].get("history", [])
                for message in history:
                    db_manager.add_chat_message(session_id, message["role"], message["content"])
                
                # Remove from in-memory storage
                del _session_memory[session_id]
        
        return success
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return False

def is_authenticated(session_id: str) -> bool:
    """Check if a session is authenticated"""
    return db_manager.is_authenticated(session_id)

def cleanup_old_data(days_old: int = 7) -> int:
    """Clean up old sessions and data"""
    return db_manager.cleanup_old_sessions(days_old) 