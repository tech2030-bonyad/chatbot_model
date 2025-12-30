#!/usr/bin/env python3
"""
Standalone ChromaDB query script - runs in subprocess to avoid Windows segfault issues.
Usage: python chromadb_query.py "your question here"
Returns: JSON array of document chunks
"""
import sys
import json
import os
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
env_path = script_dir / ".env"

from dotenv import load_dotenv
load_dotenv(dotenv_path=env_path, override=True)

from openai import OpenAI
from chromadb import PersistentClient

# Config
DB_NAME = str(script_dir / "preprocessed_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 20

def query_chromadb(question: str) -> list[dict]:
    """Query ChromaDB and return results as list of dicts"""
    try:
        # Initialize OpenAI with API key
        api_key = os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key)
        
        # Create embedding
        embedding = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, 
            input=[question]
        ).data[0].embedding
        
        # Initialize ChromaDB
        chroma_client = PersistentClient(path=DB_NAME)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        
        # Query
        results = collection.query(
            query_embeddings=[embedding], 
            n_results=RETRIEVAL_K
        )
        
        # Format results
        chunks = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0] or [{}] * len(docs)
            for doc, meta in zip(docs, metas):
                chunks.append({
                    "content": doc,
                    "metadata": meta if meta else {}
                })
        
        return chunks
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps([]))
        sys.exit(0)
    
    question = sys.argv[1]
    results = query_chromadb(question)
    print(json.dumps(results))

