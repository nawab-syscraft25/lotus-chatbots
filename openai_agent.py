# openai_agent.py

import os
import openai
import asyncio
from tools.tool_registry import tool_registry
import re
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

LOTUS_SYSTEM_PROMPT = (
    "You are Lotus, the official AI assistant for Lotus Electronics. "
    "You help users with product questions, store information, order status, and shopping assistance for Lotus Electronics. "
    "Always answer as a helpful, knowledgeable, and friendly Lotus Electronics representative. "
    "If a user asks about products, use the available tools to search the Lotus Electronics catalog. "
    "Do not mention other retailers or suggest shopping elsewhere. "
    "Do not give the response in Markdown format. "
    "Instructions: "
    "- Output strict JSON with keys: 'answer', 'products', 'end', and optionally 'comparison' if the user requests a comparison or if it would help them decide. "
    "- If a comparison is included, provide a clear, concise comparison of up to 3 products, focusing on features, price, and unique selling points. "
    "- Include up to 3–4 products with name, link, price (₹), image, 3–4 features. "
    "- Tailor 'answer' to query. "
    "- If no exact product match, suggest closest available alternatives and mention stock status if helpful. "
    "- Skip incomplete items. "
    "- Be very intelligent in product selection and recommendations. Don't show same products again. "
    "- In features always try to show the unique features of the product then other features. "
    "- No markdown, no extra text. JSON only. "
    "- Always try to infer what the customer is looking for, such as their budget, room size, preferred brand, or specific features. "
    "- If the user's request is ambiguous or incomplete, ask clarifying questions to better understand their needs before making recommendations. "
    "- After providing recommendations, ask a relevant follow-up question to help the user make a decision or clarify their needs. "
    "- If the user provides a phone number and it is valid, automatically proceed to send an OTP (or sign in if password is provided), and clearly guide the user to the next authentication step. Do not just thank the user for providing their phone number. "
    "- Never say an OTP has been sent unless you have actually called the send_otp tool and received a successful response. "
    "- Always use the send_otp tool to send an OTP before telling the user that an OTP has been sent. "
    "- If the user requests, or if it would help, offer to compare products and include a 'comparison' key in the JSON response. "
    "- If the user is authenticated (auth_token is present in session), and they ask to view their orders, always use the get_orders tool to fetch and display their orders. "
    "- Do not ask the user to log in or provide their password again if they are already authenticated. "
    "- Only prompt for login or OTP if the user is not authenticated. "
    "- After authentication, use the auth_token for all further tool calls that require authentication. "
    "Guide the user through the buying process as a helpful Lotus Electronics representative. "
    "Respond with: "
    "{"
    "  \"status\": \"success\","
    "  \"data\": {"
    "    \"answer\": \"...\","
    "    \"products\": [{\"name\": \"...\", \"link\": \"...\", \"price\": \"₹...\", \"image\": \"...\", \"features\": [\"...\"]}],"
    "    \"comparison\": [{\"name\": \"...\", \"vs_name\": \"...\", \"differences\": [\"...\"]}],"
    "    \"end\": \"...\""
    "  }"
    "}"
)

def extract_json_from_response(text):
    """Extract JSON from LLM response, handling extra text"""
    try:
        # Try to parse as direct JSON first
        return json.loads(text)
    except:
        # Look for JSON object in the text
        match = re.search(r'({.*})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
    # Return error response if JSON extraction fails
    return {
        "status": "error",
        "data": {
            "answer": "Sorry, I could not process the response properly.",
            "products": [],
            "end": ""
        }
    }

async def chat_with_agent(message: str, session_id: str, memory: dict):
    """
    Orchestrate chat with OpenAI GPT-4o, handle tool-calling, and manage memory.
    """
    # Prepare conversation history for OpenAI
    history = memory.get("history", [])
    # Always start with the system prompt
    messages = ([{"role": "system", "content": LOTUS_SYSTEM_PROMPT}] if not history or history[0].get("role") != "system" else []) + history + [{"role": "user", "content": message}]

    # Prepare function schemas for OpenAI
    function_schemas = [schema for _, schema in tool_registry.values()]

    # Call OpenAI with function-calling enabled
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=function_schemas,
        function_call="auto"
    )
    reply = response.choices[0].message

    # If the LLM wants to call a function/tool
    if reply.function_call:
        func_name = reply.function_call.name
        func_args = reply.function_call.arguments
        tool_func, _ = tool_registry.get(func_name, (None, None))
        if tool_func is None:
            tool_response = {"error": f"Tool '{func_name}' not found."}
        else:
            try:
                parsed_args = json.loads(func_args) if isinstance(func_args, str) else func_args
                # Await async tools
                if asyncio.iscoroutinefunction(tool_func):
                    tool_response = await tool_func(**parsed_args)
                else:
                    tool_response = tool_func(**parsed_args)
            except Exception as e:
                tool_response = {"error": str(e)}
        # Add tool call and result to messages
        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": func_name,
                "arguments": func_args
            }
        })
        messages.append({
            "role": "function",
            "name": func_name,
            "content": str(tool_response)
        })
        # Call LLM again to generate the final response
        response2 = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        final_reply = response2.choices[0].message.content
        # Extract JSON from the response
        final_json = extract_json_from_response(final_reply)
        # Update memory
        memory["history"] = messages + [{"role": "assistant", "content": final_reply}]
        if func_name in ["verify_otp", "sign_in"] and tool_response.get("error") == "0":
            auth_token = (
                tool_response.get("auth_token") or
                (tool_response.get("data", {}).get("auth_token") if isinstance(tool_response.get("data"), dict) else None)
            )
            if auth_token:
                memory["auth_token"] = auth_token
                memory["phone"] = parsed_args.get("phone")
                memory["is_authenticated"] = True
                if "data" in tool_response:
                    memory["user_data"] = tool_response["data"]
        return final_json
    else:
        # Normal chat, no tool call - extract JSON from response
        final_json = extract_json_from_response(reply.content)
        # Update memory
        memory["history"] = messages + [{"role": "assistant", "content": reply.content}]
        return final_json

get_orders_schema = {
    "name": "get_orders",
    "description": (
        "Fetch the user's completed orders from Lotus Electronics. "
        "Use this tool whenever the user asks to see, view, or check their orders, order history, or past purchases, "
        "and the user is authenticated (auth_token is present in session). "
        "Do not ask the user to log in again if already authenticated."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "auth_token": {"type": "string"}
        },
        "required": ["auth_token"]
    }
}


