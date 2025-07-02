import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

class DatabaseManager:
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone TEXT UNIQUE NOT NULL,
                    auth_token TEXT,
                    user_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id INTEGER,
                    auth_token TEXT,
                    phone TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create chat_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.commit()
    
    def create_or_update_user(self, phone: str, auth_token: str, user_data: Dict = None) -> int:
        """Create or update a user record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE phone = ?", (phone,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                # Update existing user
                cursor.execute('''
                    UPDATE users 
                    SET auth_token = ?, user_data = ?, last_login = CURRENT_TIMESTAMP
                    WHERE phone = ?
                ''', (auth_token, json.dumps(user_data) if user_data else None, phone))
                return existing_user[0]
            else:
                # Create new user
                cursor.execute('''
                    INSERT INTO users (phone, auth_token, user_data)
                    VALUES (?, ?, ?)
                ''', (phone, auth_token, json.dumps(user_data) if user_data else None))
                return cursor.lastrowid
    
    def create_session(self, session_id: str, user_id: int = None, auth_token: str = None, phone: str = None) -> bool:
        """Create a new session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions (session_id, user_id, auth_token, phone)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, user_id, auth_token, phone))
                return True
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def update_session_auth(self, session_id: str, user_id: int, auth_token: str, phone: str) -> bool:
        """Update session with authentication data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions 
                    SET user_id = ?, auth_token = ?, phone = ?, last_activity = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (user_id, auth_token, phone, session_id))
                return True
        except Exception as e:
            print(f"Error updating session auth: {e}")
            return False
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Get session data including user info if authenticated"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.session_id, s.user_id, s.auth_token, s.phone, 
                       u.user_data, s.last_activity
                FROM sessions s
                LEFT JOIN users u ON s.user_id = u.id
                WHERE s.session_id = ?
            ''', (session_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    "session_id": result[0],
                    "user_id": result[1],
                    "auth_token": result[2],
                    "phone": result[3],
                    "user_data": json.loads(result[4]) if result[4] else None,
                    "last_activity": result[5]
                }
            return None
    
    def is_authenticated(self, session_id: str) -> bool:
        """Check if a session is authenticated"""
        session_data = self.get_session_data(session_id)
        return session_data is not None and session_data.get("auth_token") is not None
    
    def add_chat_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a chat message to history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_history (session_id, role, content)
                    VALUES (?, ?, ?)
                ''', (session_id, role, content))
                return True
        except Exception as e:
            print(f"Error adding chat message: {e}")
            return False
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT role, content, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                })
            
            # Return in chronological order
            return list(reversed(history))
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update last activity timestamp for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions 
                    SET last_activity = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (session_id,))
                return True
        except Exception as e:
            print(f"Error updating session activity: {e}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old sessions and their chat history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old chat history
                cursor.execute('''
                    DELETE FROM chat_history 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE last_activity < datetime('now', '-{} days')
                    )
                '''.format(days_old))
                
                # Delete old sessions
                cursor.execute('''
                    DELETE FROM sessions 
                    WHERE last_activity < datetime('now', '-{} days')
                '''.format(days_old))
                
                return cursor.rowcount
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return 0

# Global database instance
db_manager = DatabaseManager() 