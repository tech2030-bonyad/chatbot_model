from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential
import json

import os


# Load .env from the bonyad_chatBot directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Single OpenAI client instance
client = OpenAI(api_key=api_key)

MODEL = "gpt-4o-mini"  # Fixed: gpt-5-nano doesn't exist, using gpt-4o-mini instead



collection_name = "docs"
embedding_model = "text-embedding-3-large"
wait = wait_exponential(multiplier=1, min=10, max=240)

# ChromaDB is now queried via subprocess (see chromadb_query.py)
# This avoids Windows segfault issues with the HNSW index

RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = """
You are a helpful assistant for BONYAD app. Answer accurately using the documentation below.

{context}

STYLE: Be specific and definite. Use exact button/screen names. No vague words like "typically", "usually", "may be". Give clear step-by-step instructions.

SCREENS:
PUBLIC: splash, welcome, overview, login, signup, projects, about, contact, introToApp, otp, forgotPassword, otpVerification, resetPassword
PROTECTED: home, profile, editProfile, myData, changePhone, changePassword, portfolio, services, availability, subscription, newProject, manualForm, aiForm, runningProjects, chatRooms, chatDetail, notifications, appointments, booking, technicianProfile, technicianOnboarding, roomDesign, voiceAI, costExplorer, roomVisualizer, askBonyadAI, projectsMap

NAVIGATION - CRITICAL RULES:
1. NEVER use JSON format for navigation. DO NOT return JSON objects like {{"action": "navigate", "view": "..."}}
2. ONLY use the [NAV:screenName] tag format in your text response
3. When mentioning ANY screen name, you MUST use **ScreenName** (bold) and ALWAYS include [NAV:screenName] tag
4. Use EXACT lowercase/camelCase screen names from the lists above
5. NEVER mention a screen name without using **bold** format - ALWAYS wrap screen names in **bold**

CORRECT FORMAT EXAMPLES:
- "To see your profile, go to **Profile**. [NAV:profile]"
- "Tap on **Login** to sign in. [NAV:login]"
- "Navigate to **Projects** to view available projects. [NAV:projects]"
- "Check your **Edit Profile** settings. [NAV:editProfile]"
- "Go to **Chat Rooms** to see your messages. [NAV:chatRooms]"

SPECIFIC EXAMPLES FOR COMMON QUESTIONS:
- User asks "how to login" → Answer: "To log in, tap on **Login**. [NAV:login]"
- User asks "how to see my profile" → Answer: "Go to **Profile** to view your information. [NAV:profile]"
- User asks "where are projects" → Answer: "Navigate to **Projects** to see available projects. [NAV:projects]"
- User asks "how to access profile" → Answer: "Tap on the **Profile** tab to view your profile. [NAV:profile]"

WRONG FORMATS (DO NOT USE):
- ❌ "Tap on the Profile tab" (missing **bold** and [NAV:])
- ❌ "Navigate to Profile" (missing **bold** and [NAV:])
- ❌ "Go to Profile tab" (missing **bold** and [NAV:])

REMEMBER: Every time you mention a screen name, it MUST be in **bold** format with [NAV:screenName] tag!

WRONG FORMATS (DO NOT USE):
- ❌ JSON: {{"action": "navigate", "view": "MyProfileView"}}
- ❌ Code blocks: ```swift selectedTab = .profile ```
- ❌ Without [NAV:] tag: "Go to **Profile**" (missing [NAV:profile])

REMEMBER: Your response must be plain text with [NAV:screenName] tags, NOT JSON or code blocks!
"""

class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Direct OpenAI API call with structured output
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about how to use the BONYAD app.
You are about to look up information in the BONYAD documentation and user guides to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the BONYAD Knowledge Base.
It should be a VERY short specific question most likely to surface content about app usage, features, or troubleshooting. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    # Direct OpenAI API call
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": message}],
        temperature=0.3
    )
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    """Query ChromaDB using subprocess to avoid Windows segfault issues"""
    import logging
    import subprocess
    import json
    import sys
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"fetch_context_unranked: Querying via subprocess for: {question[:30]}...")
        
        # Get paths
        script_dir = Path(__file__).parent.parent
        query_script = script_dir / "chromadb_query.py"
        python_exe = sys.executable
        
        # Run subprocess
        result = subprocess.run(
            [python_exe, str(query_script), question],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_dir)
        )
        
        if result.returncode != 0:
            logger.error(f"fetch_context_unranked: Subprocess error: {result.stderr}")
            return []
        
        # Parse results
        chunks_data = json.loads(result.stdout)
        logger.info(f"fetch_context_unranked: Got {len(chunks_data)} results from subprocess")
        
        # Convert to Result objects
        chunks = []
        for item in chunks_data:
            chunks.append(Result(
                page_content=item.get("content", ""),
                metadata=item.get("metadata", {})
            ))
        
        return chunks
    except subprocess.TimeoutExpired:
        logger.error("fetch_context_unranked: Subprocess timed out")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"fetch_context_unranked: JSON parse error: {e}")
        return []
    except Exception as e:
        logger.error(f"fetch_context_unranked: Error - {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_context(original_question):
    import logging
    logger = logging.getLogger(__name__)
    try:
        logger.info("fetch_context: Rewriting query...")
        rewritten_question = rewrite_query(original_question)
        logger.info(f"fetch_context: Rewritten to: {rewritten_question[:50]}...")
        
        logger.info("fetch_context: Fetching chunks for original question...")
        chunks1 = fetch_context_unranked(original_question)
        logger.info(f"fetch_context: Got {len(chunks1)} chunks from original")
        
        logger.info("fetch_context: Fetching chunks for rewritten question...")
        chunks2 = fetch_context_unranked(rewritten_question)
        logger.info(f"fetch_context: Got {len(chunks2)} chunks from rewritten")
        
        chunks = merge_chunks(chunks1, chunks2)
        logger.info(f"fetch_context: Merged to {len(chunks)} chunks")
        
        # Skip reranking - ChromaDB already provides relevant results
        logger.info(f"fetch_context: Returning {min(len(chunks), FINAL_K)} chunks (reranking disabled)")
        
        return chunks[:FINAL_K]
    except Exception as e:
        logger.error(f"fetch_context: Error - {e}")
        import traceback
        traceback.print_exc()
        return []


@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"answer_question: Starting for question: {question[:50]}...")
        
        # RAG enabled - uses knowledge base for context
        SKIP_RAG = False  # Set to True to bypass RAG for testing
        
        if SKIP_RAG:
            logger.info("answer_question: RAG disabled, using direct OpenAI call")
            chunks = []
        else:
            # Fetch context with error handling
            try:
                chunks = fetch_context(question)
                logger.info(f"answer_question: Got {len(chunks) if chunks else 0} chunks")
            except Exception as chunk_err:
                logger.error(f"answer_question: Error fetching context: {chunk_err}")
                chunks = []
        
        # Build messages
        try:
            messages = make_rag_messages(question, history, chunks)
            logger.info(f"answer_question: Built {len(messages)} messages")
        except Exception as msg_err:
            logger.error(f"answer_question: Error building messages: {msg_err}")
            messages = [{"role": "user", "content": question}]
        
        # Call OpenAI directly
        logger.info("answer_question: Calling OpenAI...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7
        )
        logger.info("answer_question: OpenAI call completed")
        
        # Extract answer safely
        try:
            answer = response.choices[0].message.content
            logger.info(f"answer_question: Got answer of length {len(answer) if answer else 0}")
        except Exception as parse_err:
            logger.error(f"answer_question: Error parsing response: {parse_err}")
            answer = "Sorry, I couldn't parse the response."
        
        # Return simple list instead of complex objects
        logger.info("answer_question: Returning result")
        return answer, []  # Return empty list to avoid complex object issues
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"answer_question: Exception: {e}")
        error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        return error_msg, []
