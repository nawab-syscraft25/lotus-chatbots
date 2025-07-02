# Lotus Shopping Assistant with Database Authentication

A chatbot application for Lotus Electronics with persistent storage for authenticated users.

## Features

- **Hybrid Storage System**: 
  - Anonymous users: In-memory storage (temporary)
  - Authenticated users: SQLite database storage (persistent)
- **User Authentication**: OTP-based authentication via Lotus Electronics API
- **Session Management**: Persistent sessions for authenticated users
- **Chat History**: Complete conversation history stored in database
- **Database Management**: Tools for maintenance and monitoring

## Database Schema

### Users Table
- `id`: Primary key
- `phone`: User's phone number (unique)
- `auth_token`: Authentication token from Lotus API
- `user_data`: JSON data from user profile
- `created_at`: Account creation timestamp
- `last_login`: Last login timestamp

### Sessions Table
- `id`: Primary key
- `session_id`: Unique session identifier
- `user_id`: Foreign key to users table
- `auth_token`: Current session auth token
- `phone`: User's phone number
- `created_at`: Session creation timestamp
- `last_activity`: Last activity timestamp

### Chat History Table
- `id`: Primary key
- `session_id`: Foreign key to sessions table
- `role`: Message role (user/assistant)
- `content`: Message content
- `timestamp`: Message timestamp

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

### Authentication
- `POST /auth/check-user` - Check if user exists
- `POST /auth/send-otp` - Send OTP to phone
- `POST /auth/verify-otp` - Verify OTP and authenticate
- `POST /auth/sign-in` - Sign in with password
- `GET /auth/status/{session_id}` - Check authentication status

### Chat
- `POST /chat` - Send message to chatbot

## Database Management

Use the `manage_db.py` script for database maintenance:

```bash
# Show database statistics
python manage_db.py stats

# Clean up old sessions (default: 7 days)
python manage_db.py cleanup --days 7

# Show recent users
python manage_db.py users --limit 10

# Show session details
python manage_db.py session --session-id <session_id>
```

## How It Works

1. **Anonymous Users**: 
   - Start with in-memory storage
   - Chat history is temporary
   - Data is lost when session ends

2. **Authentication Process**:
   - User provides phone number
   - OTP is sent via Lotus API
   - User verifies OTP
   - Session is migrated to database storage
   - Previous chat history is preserved

3. **Authenticated Users**:
   - All data stored in SQLite database
   - Persistent chat history
   - Session survives browser restarts
   - Access to order history and personalized features

## Security Features

- Only authenticated users get persistent storage
- Session activity tracking
- Automatic cleanup of old sessions
- Secure token storage
- CORS protection

## File Structure

```
├── app.py                 # Main FastAPI application
├── memory/
│   ├── database.py        # Database manager
│   ├── memory_store.py    # Hybrid memory system
│   └── __init__.py
├── tools/                 # API tools and authentication
├── templates/             # HTML templates
├── static/                # CSS/JS files
├── manage_db.py           # Database management script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Database File

The SQLite database is automatically created as `chatbot.db` in the project root when the application starts.

## Maintenance

- **Automatic Cleanup**: Old sessions are automatically cleaned up
- **Manual Cleanup**: Use `manage_db.py cleanup` for manual cleanup
- **Backup**: The `chatbot.db` file can be backed up directly
- **Monitoring**: Use `manage_db.py stats` to monitor database usage 