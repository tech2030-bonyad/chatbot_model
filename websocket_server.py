"""
WebSocket Server for BONYAD ChatBot
Connects the React Native frontend with the Python RAG chatbot backend
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pro_implementation.answer import answer_question

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="BONYAD ChatBot WebSocket Server")

# Enable CORS for all origins (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager to handle multiple WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_histories: Dict[str, list] = {}
        self.users_seen_welcome: Set[str] = set()  # Track users who have seen welcome message
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        logger.info(f"User {user_id} disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
    
    async def send_typing_indicator(self, user_id: str, is_typing: bool):
        """Send typing indicator to user"""
        await self.send_message(user_id, {
            "type": "TYPING_INDICATOR",
            "isTyping": is_typing,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    def get_history(self, user_id: str) -> list:
        """Get conversation history for a user"""
        return self.user_histories.get(user_id, [])
    
    def add_to_history(self, user_id: str, role: str, content: str):
        """Add message to user's conversation history"""
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
        self.user_histories[user_id].append({
            "role": role,
            "content": content
        })
    
    def clear_history(self, user_id: str):
        """Clear conversation history for a user"""
        self.user_histories[user_id] = []
        logger.info(f"Cleared history for user {user_id}")


manager = ConnectionManager()


# Request/Response models for REST API
class ChatRequest(BaseModel):
    userId: str
    message: str
    platform: str = "web"
    hasSeenWelcome: bool = False


class ChatResponse(BaseModel):
    type: str
    content: str
    timestamp: str
    sources: list = []


@app.get("/")
async def root():
    return {
        "service": "BONYAD ChatBot API Server",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "rest_api": "/api/chat (POST) - Stateless HTTP API for containerized deployments",
            "websocket": "/ws/chat?userId={userId} (Legacy)",
            "websocket_app": "/app/chat-bot?userId={userId} (Legacy)",
            "health": "/health"
        },
        "recommended": "Use /api/chat for production (container-friendly, stateless, scalable)"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest):
    """
    REST API endpoint for chatbot interactions
    Perfect for containerized deployments - stateless and scalable
    
    Request body:
    {
        "userId": "user-123",
        "message": "What is BONYAD?",
        "platform": "web",
        "hasSeenWelcome": false
    }
    
    Response:
    {
        "type": "BOT_MESSAGE",
        "content": "BONYAD is...",
        "timestamp": "2024-01-01T00:00:00.000Z",
        "sources": [...]
    }
    """
    try:
        user_id = request.userId
        message = request.message.strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"📥 REST API - Received from {user_id}: {message[:50]}...")
        
        # Add user message to history
        manager.add_to_history(user_id, "user", message)
        
        # Get conversation history (excluding the current message)
        history = manager.get_history(user_id)[:-1]
        
        # Get answer from RAG chatbot
        logger.info(f"🤖 Processing question for {user_id}: {message[:50]}...")
        
        try:
            # Call the RAG system synchronously
            result = answer_question(message, history)
            
            # Safely unpack the result
            if isinstance(result, tuple) and len(result) >= 2:
                answer, chunks = result[0], result[1]
            else:
                answer = str(result) if result else "Sorry, I couldn't generate a response."
                chunks = []
            
            logger.info(f"✅ Generated response: {len(answer)} chars, {len(chunks) if chunks else 0} sources")
            
        except Exception as e:
            logger.error(f"❌ Error in answer_question: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
        
        # Add bot response to history
        manager.add_to_history(user_id, "assistant", answer)
        
        # Extract sources from chunks
        sources = []
        if chunks:
            for chunk in chunks[:3]:
                try:
                    if hasattr(chunk, 'metadata'):
                        sources.append({
                            "title": chunk.metadata.get('title', 'Unknown'),
                            "content": chunk.page_content[:200] if hasattr(chunk, 'page_content') else '',
                        })
                except Exception as e:
                    logger.warning(f"Could not extract source: {e}")
        
        # Return response
        response = ChatResponse(    
            type="BOT_MESSAGE",
            content=answer,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sources=sources
        )
        
        logger.info(f"📤 REST API - Sent response to {user_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in chat_api: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/chat/history/{user_id}")
async def get_chat_history(user_id: str):
    """
    Get conversation history for a user
    
    Path parameter:
    - user_id: The user ID whose history should be retrieved
    
    Response:
    {
        "userId": "user-123",
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            ...
        ],
        "messageCount": 2
    }
    
    Example:
    GET http://localhost:8080/api/chat/history/user-123
    """
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="userId is required")
        
        logger.info(f"📖 Getting history for user {user_id}")
        
        # Get history from server
        history = manager.get_history(user_id)
        
        logger.info(f"✅ Retrieved {len(history)} messages for user {user_id}")
        
        return {
            "userId": user_id,
            "history": history,
            "messageCount": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting history: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/api/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """
    Clear conversation history for a user
    
    Path parameter:
    - user_id: The user ID whose history should be cleared
    
    Response:
    {
        "message": "History cleared successfully",
        "userId": "user-123"
    }
    
    Example:
    DELETE http://localhost:8080/api/chat/history/user-123
    """
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="userId is required")
        
        logger.info(f"🗑️ Clearing history for user {user_id}")
        
        # Clear history on server
        manager.clear_history(user_id)
        
        logger.info(f"✅ History cleared for user {user_id}")
        
        return {
            "message": "History cleared successfully",
            "userId": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error clearing history: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def handle_websocket_connection(websocket: WebSocket):
    """
    Shared WebSocket connection handler
    Accepts query parameters:
    - userId: unique user identifier
    - hasSeenWelcome: "true" or "false" - indicates if user has seen welcome message
    
    Message format from frontend:
    {
        "type": "USER_MESSAGE",
        "content": "message text",
        "userId": "user123",
        "timestamp": "ISO-8601 string"
    }
    
    Message format to frontend:
    {
        "type": "BOT_MESSAGE" | "TYPING_INDICATOR" | "ERROR",
        "content": "message text",
        "timestamp": "ISO-8601 string"
    }
    """
    userId = websocket.query_params.get("userId", f"user-{datetime.utcnow().timestamp()}")
    has_seen_welcome = websocket.query_params.get("hasSeenWelcome", "false").lower() == "true"
    
    await manager.connect(websocket, userId)
    
    # Send welcome message only if user hasn't seen it before
    if not has_seen_welcome:
        await manager.send_message(userId, {
            "type": "BOT_MESSAGE",
            "content": "مرحباً! أنا مساعد بُنْياد. كيف يمكنني مساعدتك اليوم؟\n\nHello! I'm the BONYAD Assistant. How can I help you today?",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        logger.info(f"Sent welcome message to user {userId} (first time)")
    else:
        logger.info(f"User {userId} has already seen welcome message - skipping")
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_json()
            logger.info(f"Received from {userId}: {data.get('type')} - {data.get('content', '')[:50]}...")
            
            message_type = data.get("type")
            content = data.get("content", "").strip()
            
            if message_type == "USER_MESSAGE" and content:
                # Add user message to history
                manager.add_to_history(userId, "user", content)
                
                # Send typing indicator
                await manager.send_typing_indicator(userId, True)
                
                try:
                    # Get conversation history (excluding the current message)
                    history = manager.get_history(userId)[:-1]  # Exclude current message
                    
                    # Get answer from chatbot
                    logger.info(f"Processing question for {userId}: {content[:50]}...")
                    
                    # TEST MODE: Use simple response first to test WebSocket
                    USE_SIMPLE_RESPONSE = False  # Set to False to use real RAG
                    
                    if USE_SIMPLE_RESPONSE:
                        logger.info("Using simple test response")
                        answer = f"Test response to: {content}\n\nThis is a test message to verify WebSocket is working."
                        chunks = []
                    else:
                        try:
                            # Call synchronously - ChromaDB crashes with thread pool executors
                            logger.info("Calling answer_question synchronously...")
                            result = answer_question(content, history)
                            logger.info(f"Got result from answer_question: type={type(result)}")
                            
                            # Safely unpack the result
                            if isinstance(result, tuple) and len(result) >= 2:
                                answer, chunks = result[0], result[1]
                            else:
                                answer = str(result) if result else "Sorry, I couldn't generate a response."
                                chunks = []
                            
                            logger.info(f"Answer length: {len(answer) if answer else 0}, Chunks: {len(chunks) if chunks else 0}")
                        except asyncio.TimeoutError:
                            logger.error("answer_question timed out after 60 seconds")
                            answer = "Sorry, the request timed out. Please try again."
                            chunks = []
                        except Exception as answer_error:
                            logger.error(f"Error in answer_question: {answer_error}")
                            import traceback
                            traceback.print_exc()
                            answer = f"Sorry, an error occurred: {str(answer_error)}"
                            chunks = []
                    
                    # Add bot response to history
                    logger.info(f"Adding response to history for {userId}")
                    manager.add_to_history(userId, "assistant", answer)
                    logger.info(f"Added response to history for {userId}")
                    
                    # Stop typing indicator
                    logger.info(f"Stopping typing indicator for {userId}")
                    await manager.send_typing_indicator(userId, False)
                    logger.info(f"Typing indicator stopped for {userId}")
                    
                    # Send bot response
                    logger.info(f"Preparing to send response to {userId}...")
                    try:
                        # Safely extract sources from chunks
                        sources = []
                        if chunks:
                            for chunk in chunks[:3]:
                                try:
                                    if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                                        sources.append(chunk.metadata.get("source", "unknown"))
                                    else:
                                        sources.append("unknown")
                                except:
                                    sources.append("unknown")
                        
                        await manager.send_message(userId, {
                            "type": "BOT_MESSAGE",
                            "content": answer,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "metadata": {
                                "chunks_used": len(chunks) if chunks else 0,
                                "sources": sources
                            }
                        })
                        logger.info(f"Sent response to {userId}")
                    except Exception as send_error:
                        logger.error(f"Error sending response: {send_error}")
                        # Fallback: send without metadata
                        await manager.send_message(userId, {
                            "type": "BOT_MESSAGE",
                            "content": answer,
                            "timestamp": datetime.utcnow().isoformat() + "Z"
                        })
                        logger.info(f"Sent response (fallback) to {userId}")
                    
                except Exception as e:
                    logger.error(f"Error processing message for {userId}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Stop typing indicator
                    await manager.send_typing_indicator(userId, False)
                    
                    # Send error message
                    await manager.send_message(userId, {
                        "type": "BOT_MESSAGE",
                        "content": "عذراً، حدث خطأ أثناء معالجة رسالتك. يرجى المحاولة مرة أخرى.\n\nSorry, an error occurred while processing your message. Please try again.",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
            
            elif message_type == "CLEAR_HISTORY":
                # Clear conversation history
                manager.clear_history(userId)
                await manager.send_message(userId, {
                    "type": "SYSTEM_MESSAGE",
                    "content": "تم مسح المحادثة.\n\nConversation cleared.",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
            
            elif message_type == "PING":
                # Respond to ping to keep connection alive
                await manager.send_message(userId, {
                    "type": "PONG",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(userId)
        logger.info(f"WebSocket disconnected for user {userId}")
    
    except Exception as e:
        logger.error(f"Unexpected error for user {userId}: {e}")
        import traceback
        traceback.print_exc()
        manager.disconnect(userId)


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat at /ws/chat"""
    await handle_websocket_connection(websocket)


@app.websocket("/app/chat-bot")
async def websocket_endpoint_app(websocket: WebSocket):
    """WebSocket endpoint for chat at /app/chat-bot"""
    await handle_websocket_connection(websocket)


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    logger.info("🚀 Starting BONYAD ChatBot WebSocket Server...")
    logger.info("📱 React Native apps can connect to: ws://localhost:8080/ws/chat?userId={userId}")
    logger.info("🌐 Web apps can connect to: ws://localhost:8080/app/chat-bot?userId={userId}")
    logger.info("🔗 Health check: http://localhost:8080/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    )

