"""
MQTT Server for BONYAD ChatBot
Replaces WebSocket with MQTT for real-time communication
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict
import paho.mqtt.client as mqtt
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

# MQTT Configuration
# USING PUBLIC TEST BROKER (no installation needed)
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883  # Use 1883 for non-websocket connection from Python
MQTT_KEEPALIVE = 60

# LOCAL MOSQUITTO (uncomment after installing):
# MQTT_BROKER = "localhost"
# MQTT_PORT = 8083

# User state management
class ChatBotManager:
    def __init__(self):
        self.user_histories: Dict[str, list] = {}
        self.users_seen_welcome: set = set()
    
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
        logger.info(f"Added {role} message to history for user {user_id}")
    
    def clear_history(self, user_id: str):
        """Clear conversation history for a user"""
        self.user_histories[user_id] = []
        logger.info(f"Cleared history for user {user_id}")
    
    def mark_welcome_seen(self, user_id: str):
        """Mark that user has seen welcome message"""
        self.users_seen_welcome.add(user_id)


manager = ChatBotManager()
mqtt_client = None


def publish_message(user_id: str, message: dict):
    """Publish message to user's receive topic"""
    topic = f"chatbot/user/{user_id}/receive"
    payload = json.dumps(message)
    
    try:
        result = mqtt_client.publish(topic, payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"📤 Published to {topic}: {message.get('type')}")
        else:
            logger.error(f"❌ Failed to publish to {topic}: {result.rc}")
    except Exception as e:
        logger.error(f"❌ Error publishing message: {e}")


def send_typing_indicator(user_id: str, is_typing: bool):
    """Send typing indicator to user"""
    publish_message(user_id, {
        "type": "TYPING_INDICATOR",
        "isTyping": is_typing,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


def send_welcome_message(user_id: str):
    """Send welcome message to user"""
    publish_message(user_id, {
        "type": "BOT_MESSAGE",
        "content": "مرحباً! أنا مساعد بُنْياد. كيف يمكنني مساعدتك اليوم؟\n\nHello! I'm the BONYAD Assistant. How can I help you today?",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    manager.mark_welcome_seen(user_id)
    logger.info(f"Sent welcome message to user {user_id}")


def process_user_message(user_id: str, content: str):
    """Process user message and generate AI response"""
    logger.info(f"Processing message from {user_id}: {content[:50]}...")
    
    # Add user message to history
    manager.add_to_history(user_id, "user", content)
    
    # Send typing indicator
    send_typing_indicator(user_id, True)
    
    try:
        # Get conversation history (excluding current message)
        history = manager.get_history(user_id)[:-1]
        
        # Get answer from chatbot
        logger.info(f"Calling answer_question for {user_id}...")
        result = answer_question(content, history)
        
        # Safely unpack the result
        if isinstance(result, tuple) and len(result) >= 2:
            answer, chunks = result[0], result[1]
        else:
            answer = str(result) if result else "Sorry, I couldn't generate a response."
            chunks = []
        
        logger.info(f"Answer length: {len(answer)}, Chunks: {len(chunks)}")
        
        # Add bot response to history
        manager.add_to_history(user_id, "assistant", answer)
        
        # Stop typing indicator
        send_typing_indicator(user_id, False)
        
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
        
        # Send bot response
        publish_message(user_id, {
            "type": "BOT_MESSAGE",
            "content": answer,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": {
                "chunks_used": len(chunks) if chunks else 0,
                "sources": sources
            }
        })
        
        logger.info(f"✅ Sent response to user {user_id}")
        
    except Exception as e:
        logger.error(f"❌ Error processing message for {user_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Stop typing indicator
        send_typing_indicator(user_id, False)
        
        # Send error message
        publish_message(user_id, {
            "type": "BOT_MESSAGE",
            "content": "عذراً، حدث خطأ أثناء معالجة رسالتك. يرجى المحاولة مرة أخرى.\n\nSorry, an error occurred while processing your message. Please try again.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })


def handle_message(user_id: str, message_data: dict):
    """Handle incoming message from user"""
    message_type = message_data.get("type")
    
    if message_type == "CONNECT":
        # Handle initial connection
        has_seen_welcome = message_data.get("hasSeenWelcome", False)
        logger.info(f"User {user_id} connected, hasSeenWelcome={has_seen_welcome}")
        
        if not has_seen_welcome:
            send_welcome_message(user_id)
        else:
            logger.info(f"User {user_id} has already seen welcome - skipping")
    
    elif message_type == "USER_MESSAGE":
        content = message_data.get("content", "").strip()
        if content:
            process_user_message(user_id, content)
    
    elif message_type == "CLEAR_HISTORY":
        manager.clear_history(user_id)
        publish_message(user_id, {
            "type": "SYSTEM_MESSAGE",
            "content": "تم مسح المحادثة.\n\nConversation cleared.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    elif message_type == "PING":
        publish_message(user_id, {
            "type": "PONG",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })


def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        logger.info("✅ Connected to MQTT broker")
        
        # Subscribe to all user send topics
        topic = "chatbot/user/+/send"
        client.subscribe(topic, qos=1)
        logger.info(f"✅ Subscribed to: {topic}")
        
        # Subscribe to all user connect topics
        connect_topic = "chatbot/user/+/connect"
        client.subscribe(connect_topic, qos=1)
        logger.info(f"✅ Subscribed to: {connect_topic}")
        
    else:
        logger.error(f"❌ Failed to connect to MQTT broker. Return code: {rc}")


def on_message(client, userdata, msg):
    """Callback when message received"""
    try:
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        logger.info(f"📨 Received message on topic: {topic}")
        
        # Extract user_id from topic
        # Topic format: chatbot/user/{userId}/send or chatbot/user/{userId}/connect
        parts = topic.split('/')
        if len(parts) >= 3:
            user_id = parts[2]
            
            # Parse message
            message_data = json.loads(payload)
            
            # Handle message
            handle_message(user_id, message_data)
        else:
            logger.warning(f"⚠️ Invalid topic format: {topic}")
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON: {e}")
    except Exception as e:
        logger.error(f"❌ Error handling message: {e}")
        import traceback
        traceback.print_exc()


def on_disconnect(client, userdata, rc):
    """Callback when disconnected from MQTT broker"""
    if rc != 0:
        logger.warning(f"⚠️ Unexpected disconnection. Return code: {rc}")
        logger.info("🔄 Attempting to reconnect...")
    else:
        logger.info("🔌 Disconnected from MQTT broker")


def on_subscribe(client, userdata, mid, granted_qos):
    """Callback when subscribed to topic"""
    logger.info(f"✅ Subscription confirmed (QoS: {granted_qos})")


def main():
    """Main function to start MQTT chatbot server"""
    global mqtt_client
    
    logger.info("🚀 Starting BONYAD ChatBot MQTT Server...")
    logger.info(f"📡 MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    
    # Create MQTT client
    mqtt_client = mqtt.Client(client_id="bonyad-chatbot-server", clean_session=False)
    
    # Set callbacks
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_subscribe = on_subscribe
    
    # Connect to broker
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
        logger.info("🔗 Connecting to MQTT broker...")
        
        # Start loop
        mqtt_client.loop_forever()
        
    except KeyboardInterrupt:
        logger.info("⚠️ Keyboard interrupt received")
        mqtt_client.disconnect()
        logger.info("👋 MQTT Chatbot Server stopped")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

