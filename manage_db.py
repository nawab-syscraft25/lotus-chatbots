#!/usr/bin/env python3
"""
Database management script for the chatbot
"""

import argparse
from memory.database import db_manager
from memory.memory_store import cleanup_old_data
import sqlite3

def show_stats():
    """Show database statistics"""
    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()
        
        # Count users
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # Count sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        
        # Count authenticated sessions
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE auth_token IS NOT NULL")
        auth_session_count = cursor.fetchone()[0]
        
        # Count chat messages
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        message_count = cursor.fetchone()[0]
        
        # Get recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM sessions 
            WHERE last_activity > datetime('now', '-1 day')
        """)
        recent_sessions = cursor.fetchone()[0]
        
        print("=== Database Statistics ===")
        print(f"Total Users: {user_count}")
        print(f"Total Sessions: {session_count}")
        print(f"Authenticated Sessions: {auth_session_count}")
        print(f"Total Chat Messages: {message_count}")
        print(f"Active Sessions (last 24h): {recent_sessions}")
        print(f"Database File: {db_manager.db_path}")

def cleanup_old_sessions(days=7):
    """Clean up old sessions"""
    print(f"Cleaning up sessions older than {days} days...")
    deleted_count = cleanup_old_data(days)
    print(f"Deleted {deleted_count} old sessions")

def show_recent_users(limit=10):
    """Show recent users"""
    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT phone, created_at, last_login 
            FROM users 
            ORDER BY last_login DESC 
            LIMIT ?
        """, (limit,))
        
        print(f"\n=== Recent Users (last {limit}) ===")
        for row in cursor.fetchall():
            print(f"Phone: {row[0]}, Created: {row[1]}, Last Login: {row[2]}")

def show_session_details(session_id):
    """Show details for a specific session"""
    session_data = db_manager.get_session_data(session_id)
    if session_data:
        print(f"\n=== Session Details for {session_id} ===")
        print(f"User ID: {session_data['user_id']}")
        print(f"Phone: {session_data['phone']}")
        print(f"Authenticated: {bool(session_data['auth_token'])}")
        print(f"Last Activity: {session_data['last_activity']}")
        
        # Get chat history
        history = db_manager.get_chat_history(session_id, limit=5)
        print(f"\nRecent Messages ({len(history)}):")
        for msg in history:
            print(f"  [{msg['role']}] {msg['content'][:50]}...")
    else:
        print(f"Session {session_id} not found")

def main():
    parser = argparse.ArgumentParser(description="Database management for chatbot")
    parser.add_argument("command", choices=["stats", "cleanup", "users", "session"], 
                       help="Command to execute")
    parser.add_argument("--days", type=int, default=7, 
                       help="Days for cleanup (default: 7)")
    parser.add_argument("--limit", type=int, default=10, 
                       help="Limit for user list (default: 10)")
    parser.add_argument("--session-id", type=str, 
                       help="Session ID for session details")
    
    args = parser.parse_args()
    
    if args.command == "stats":
        show_stats()
    elif args.command == "cleanup":
        cleanup_old_sessions(args.days)
    elif args.command == "users":
        show_recent_users(args.limit)
    elif args.command == "session":
        if not args.session_id:
            print("Error: --session-id is required for session command")
            return
        show_session_details(args.session_id)

if __name__ == "__main__":
    main() 