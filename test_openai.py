"""Test OpenAI API directly to see the exact error"""
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.environ.get("OPENAI_API_KEY")
print(f"API Key present: {bool(api_key)}")

client = OpenAI(api_key=api_key)

# Test 1: Simple completion
print("\n=== Test 1: Simple completion ===")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.3
    )
    print(f"SUCCESS! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test 2: With JSON format
print("\n=== Test 2: With JSON format ===")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply with JSON: {\"message\": \"your message\"}"},
            {"role": "user", "content": "Say hello"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    print(f"SUCCESS! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n=== Tests complete ===")
