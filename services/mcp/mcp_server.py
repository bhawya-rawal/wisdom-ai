import sys
import json
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_server")

# Try importing the official MCP SDK if available
try:
    from mcp.server.fastmcp import FastMCP
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

from database.connection import get_session
from retrieval.faiss_engine import faiss_engine
from services.rag.rag_service import rag_service
from retrieval.embeddings import embedding_service

# Define standard MCP tools that we will expose
TOOLS = [
    {
        "name": "search_scriptures",
        "description": "Search Bhagavad Gita, Quran, and Bible verses using semantic search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., 'finding peace during hardship')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of matching verses to return (default: 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ask_contextual_question",
        "description": "Ask WisdomAI spiritual or emotional questions, and receive compassionate answers backed by retrieved scripture.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Your question (e.g., 'Why do we suffer?')"
                },
                "mood": {
                    "type": "string",
                    "description": "The user's mood to tailor the tone (e.g., sadness, joy)"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "retrieve_verse",
        "description": "Retrieve specific verse details from the database by its exact ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "verse_id": {
                    "type": "string",
                    "description": "The verse identifier (e.g., 'Gita_2.47' or 'Quran_94.5')"
                }
            },
            "required": ["verse_id"]
        }
    }
]

# Business logic actions mapping to the tools
def action_search_scriptures(query: str, limit: int = 5) -> str:
    try:
        q_emb = embedding_service.get_embedding(query)
        results = faiss_engine.search(q_emb, top_k=limit)
        
        formatted = []
        for r in results:
            formatted.append(f"[{r['verse_id']}] ({r['source']})\n\"{r['text']}\"\nSimilarity Score: {r['similarity_score']:.2f}")
            
        return "\n\n---\n\n".join(formatted) if formatted else "No matching verses found."
    except Exception as e:
        return f"Error searching scriptures: {str(e)}"

def action_ask_contextual_question(question: str, mood: str = "neutral") -> str:
    try:
        # Get active db session
        db_generator = get_session()
        session = next(db_generator)
        
        reply, verse_info, _ = rag_service.answer_question(
            user_query=question,
            user_id=1,  # Default system evaluation user
            mood=mood,
            history_summary="",
            session=session
        )
        
        output = f"WisdomAI Response:\n{reply}\n\nSupporting Verse Cited:\n"
        output += f"Source: {verse_info.get('source')}\n"
        output += f"ID: {verse_info.get('verse_id')}\n"
        output += f"Text: \"{verse_info.get('text')}\"\n"
        output += f"Confidence Score: {verse_info.get('confidence')}% ({verse_info.get('confidence_label')})"
        
        return output
    except Exception as e:
        return f"Error answering question: {str(e)}"

def action_retrieve_verse(verse_id: str) -> str:
    try:
        meta = faiss_engine.verse_id_to_metadata.get(verse_id)
        if meta:
            return f"[{verse_id}] ({meta['source']})\n\"{meta['text']}\"\nMood Tags: {', '.join(meta.get('mood_tags', []))}"
        return f"Verse '{verse_id}' not found in index."
    except Exception as e:
        return f"Error retrieving verse: {str(e)}"

# -----------------
# OFFICIAL SDK RUNNER
# -----------------
def run_sdk_server():
    mcp_app = FastMCP("WisdomAI")
    
    @mcp_app.tool()
    def search_scriptures(query: str, limit: int = 5) -> str:
        """Search Bhagavad Gita, Quran, and Bible verses using semantic search."""
        return action_search_scriptures(query, limit)
        
    @mcp_app.tool()
    def ask_contextual_question(question: str, mood: str = "neutral") -> str:
        """Ask WisdomAI spiritual or emotional questions, and receive compassionate answers backed by retrieved scripture."""
        return action_ask_contextual_question(question, mood)
        
    @mcp_app.tool()
    def retrieve_verse(verse_id: str) -> str:
        """Retrieve specific verse details from the database by its exact ID."""
        return action_retrieve_verse(verse_id)
        
    logger.info("Starting WisdomAI SDK-based MCP Stdio Server...")
    mcp_app.run()

# -----------------
# MANUAL JSON-RPC STDIO RUNNER (Zero-dependency fallback)
# -----------------
def run_manual_server():
    logger.info("Starting WisdomAI manual JSON-RPC stdio MCP server...")
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            
            response = {"jsonrpc": "2.0"}
            
            if method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "wisdom-ai-mcp", "version": "1.0.0"}
                }
            elif method == "tools/list":
                response["result"] = {"tools": TOOLS}
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})
                
                tool_result = ""
                if tool_name == "search_scriptures":
                    query = args.get("query")
                    limit = args.get("limit", 5)
                    tool_result = action_search_scriptures(query, limit)
                elif tool_name == "ask_contextual_question":
                    question = args.get("question")
                    mood = args.get("mood", "neutral")
                    tool_result = action_ask_contextual_question(question, mood)
                elif tool_name == "retrieve_verse":
                    verse_id = args.get("verse_id")
                    tool_result = action_retrieve_verse(verse_id)
                else:
                    response["error"] = {"code": -32601, "message": f"Method not found: {tool_name}"}
                    
                if "error" not in response:
                    response["result"] = {
                        "content": [{"type": "text", "text": tool_result}]
                    }
            else:
                response["error"] = {"code": -32601, "message": f"Method not found: {method}"}
                
            if req_id is not None:
                response["id"] = req_id
                
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            logger.error("Error handling manual MCP stdio request: %s", e)

def main():
    if SDK_AVAILABLE:
        run_sdk_server()
    else:
        run_manual_server()

if __name__ == "__main__":
    main()
