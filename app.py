from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from tools import tool_registry
from memory.memory_store import get_session_memory, set_session_memory, is_authenticated, add_chat_message
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

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     # Retrieve session memory
#     memory = get_session_memory(request.session_id)
    
#     # Add user message to history
#     add_chat_message(request.session_id, "user", request.message)
    
#     # Call LLM agent with message, memory, and tool registry
#     response = await chat_with_agent(request.message, request.session_id)
    
#     # Add bot response to history
#     if response and "data" in response and "answer" in response["data"]:
#         add_chat_message(request.session_id, "assistant", response["data"]["answer"])
    
#     return {"response": response}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat requests by orchestrating with the agent.
    The chat_with_agent function handles all memory management internally.
    """
    try:
        # Call LLM agent - it handles all memory management internally
        response = await chat_with_agent(request.message, request.session_id)
        
        return {"response": response}
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error in chat_endpoint: {str(e)}")
        
        # Return a fallback response
        error_response = {
            "status": "error",
            "data": {
                "answer": "Sorry, I encountered an error while processing your request. Please try again.",
                "products": [],
                "end": ""
            }
        }
        
        return {"response": error_response}

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("chatbot-sale.html", {"request": request})

# === Run Server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 