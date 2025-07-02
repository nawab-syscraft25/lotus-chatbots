from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from tools import tool_registry
from memory.memory_store import get_session_memory, authenticate_user, add_chat_message, is_authenticated
from openai_agent import chat_with_agent
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from tools.auth import check_user, send_otp, verify_otp, sign_in


app = FastAPI(title="Lotus Shopping Assistant")
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class AuthRequest(BaseModel):
    phone: str
    session_id: str

class OTPRequest(BaseModel):
    phone: str
    otp: str
    session_id: str

class SignInRequest(BaseModel):
    phone: str
    password: str
    session_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Retrieve session memory
    memory = get_session_memory(request.session_id)
    
    # Add user message to history
    add_chat_message(request.session_id, "user", request.message)
    
    # Call LLM agent with message, memory, and tool registry
    response = await chat_with_agent(request.message, request.session_id, memory)
    
    # Add bot response to history
    if response and "data" in response and "answer" in response["data"]:
        add_chat_message(request.session_id, "assistant", response["data"]["answer"])
    
    return {"response": response}

@app.post("/auth/check-user")
async def check_user_endpoint(request: AuthRequest):
    """Check if a user exists"""
    try:
        result = await check_user(request.phone)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": "1", "message": str(e)}, status_code=500)

@app.post("/auth/send-otp")
async def send_otp_endpoint(request: AuthRequest):
    """Send OTP to user's phone"""
    try:
        result = await send_otp(request.phone)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": "1", "message": str(e)}, status_code=500)

@app.post("/auth/verify-otp")
async def verify_otp_endpoint(request: OTPRequest):
    """Verify OTP and authenticate user"""
    try:
        result = await verify_otp(request.phone, request.otp, request.session_id)
        
        # If authentication successful, store in database
        if result.get("error") == "0":
            auth_token = (
                result.get("auth_token") or
                (result.get("data", {}).get("auth_token") if isinstance(result.get("data"), dict) else None)
            )
            if auth_token:
                # Authenticate user and store in database
                user_data = result.get("data") if isinstance(result.get("data"), dict) else None
                authenticate_user(request.session_id, request.phone, auth_token, user_data)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": "1", "message": str(e)}, status_code=500)

@app.post("/auth/sign-in")
async def sign_in_endpoint(request: SignInRequest):
    """Sign in with phone and password"""
    try:
        result = await sign_in(request.phone, request.password, request.session_id)
        
        # If authentication successful, store in database
        if result.get("error") == "0":
            auth_token = (
                result.get("auth_token") or
                (result.get("data", {}).get("auth_token") if isinstance(result.get("data"), dict) else None)
            )
            if auth_token:
                # Authenticate user and store in database
                user_data = result.get("data") if isinstance(result.get("data"), dict) else None
                authenticate_user(request.session_id, request.phone, auth_token, user_data)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": "1", "message": str(e)}, status_code=500)

@app.get("/auth/status/{session_id}")
async def auth_status_endpoint(session_id: str):
    """Check authentication status for a session"""
    try:
        authenticated = is_authenticated(session_id)
        memory = get_session_memory(session_id)
        
        return JSONResponse(content={
            "authenticated": authenticated,
            "phone": memory.get("phone") if authenticated else None,
            "user_data": memory.get("user_data") if authenticated else None
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

# === Run Server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 