import os
import time
import requests
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Configuration ----
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
SPACE_ID = os.environ.get("SPACE_ID")

if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, SPACE_ID]):
    raise RuntimeError("Missing required environment variables: DATABRICKS_HOST, DATABRICKS_TOKEN, or SPACE_ID")

# Ensure DATABRICKS_HOST has https://
if not DATABRICKS_HOST.startswith("http"):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# ---- Request/Response Models ----
class QueryRequest(BaseModel):
    user_id: str
    question: str
    session_id: Optional[str] = None
    max_wait: int = 120  # seconds

class QueryResponse(BaseModel):
    session_id: str
    answer_text: str
    sql: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[List[Any]]] = None
    row_count: int = 0
    status: str

class HealthResponse(BaseModel):
    status: str
    databricks_host: str
    space_id: str

# ---- In-Memory Session Storage (use Redis in production) ----
user_sessions: Dict[str, str] = {}

# ---- Genie Client ----
class GenieClient:
    def __init__(self, host: str, space_id: str, headers: Dict[str, str]):
        self.host = host.rstrip("/")
        self.space_id = space_id
        self.headers = headers
        self.conversation_id: Optional[str] = None

    def start_conversation(self) -> str:
        """Start a new Genie conversation"""
        url = f"{self.host}/api/2.0/genie/spaces/{self.space_id}/start-conversation"
        logger.info(f"Starting conversation at {url}")
        
        resp = requests.post(url, headers=self.headers, json={"content": "Starting conversation"})
        if resp.status_code != 200:
            logger.error(f"Failed to start conversation: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=f"Failed to start conversation: {resp.text}")
        
        data = resp.json()
        self.conversation_id = data.get("conversation_id") or data.get("conversation", {}).get("id")
        
        if not self.conversation_id:
            raise HTTPException(status_code=500, detail="No conversation_id in response")
        
        logger.info(f"Started conversation: {self.conversation_id}")
        return self.conversation_id

    def send_message(self, prompt: str) -> str:
        """Send a message to Genie"""
        if not self.conversation_id:
            self.start_conversation()
        
        url = f"{self.host}/api/2.0/genie/spaces/{self.space_id}/conversations/{self.conversation_id}/messages"
        logger.info(f"Sending message to conversation {self.conversation_id}")
        
        resp = requests.post(url, headers=self.headers, json={"content": prompt})
        if resp.status_code != 200:
            logger.error(f"Failed to send message: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=f"Failed to send message: {resp.text}")
        
        data = resp.json()
        message_id = data.get("id") or data.get("message", {}).get("id")
        
        if not message_id:
            raise HTTPException(status_code=500, detail="No message id in response")
        
        logger.info(f"Message sent: {message_id}")
        return message_id

    def get_message_status(self, message_id: str, max_wait: int = 120) -> Dict[str, Any]:
        """Poll for message completion"""
        url = f"{self.host}/api/2.0/genie/spaces/{self.space_id}/conversations/{self.conversation_id}/messages/{message_id}"
        start = time.time()
        
        while time.time() - start < max_wait:
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                logger.error(f"Status check error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Status check error: {resp.text}")
            
            data = resp.json()
            status = data.get("status")
            logger.info(f"Message status: {status}")
            
            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                return data
            
            time.sleep(3)
        
        raise HTTPException(status_code=504, detail="Genie query timed out")

    def get_query_result(self, message_id: str, attachment_id: str) -> Optional[Dict[str, Any]]:
        """Get query results from attachment"""
        url = f"{self.host}/api/2.0/genie/spaces/{self.space_id}/conversations/{self.conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
        
        try:
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to get query result: {e}")
        
        return None

    def extract_response(self, message_data: Dict[str, Any], message_id: str) -> Dict[str, Any]:
        """Extract analysis text, SQL, and tabular results from attachments"""
        result = {
            "analysis": "",
            "sql": None,
            "columns": None,
            "rows": None,
            "status": message_data.get("status"),
            "row_count": 0
        }
        
        attachments = message_data.get("attachments", [])
        
        for att in attachments:
            # Extract text content (analysis)
            if "text" in att:
                text = att["text"].get("content", "")
                result["analysis"] += f"\n{text}".strip()
            
            # Extract query (SQL + description)
            if "query" in att:
                q = att["query"]
                result["sql"] = q.get("query")
                desc = q.get("description", "")
                if desc:
                    result["analysis"] = f"{desc}\n{result['analysis']}".strip()
                
                # Get query results
                att_id = att.get("attachment_id")
                if att_id:
                    qr = self.get_query_result(message_id, att_id)
                    if qr:
                        stmt = qr.get("statement_response", {})
                        manifest = stmt.get("manifest", {})
                        schema = manifest.get("schema", {})
                        result["columns"] = [c.get("name") for c in schema.get("columns", [])]
                        
                        # Try data_array first
                        data = stmt.get("result", {}).get("data_array")
                        if data and result["columns"]:
                            result["rows"] = data
                            result["row_count"] = len(data)
                        else:
                            # Try data_typed_array
                            typed = stmt.get("result", {}).get("data_typed_array", [])
                            if typed:
                                rows = []
                                for row in typed:
                                    vals = []
                                    for v in row.get("values", []):
                                        # Extract value based on type
                                        val = (v.get("str") or v.get("int") or v.get("double") or 
                                               v.get("bool") or v.get("date") or v.get("timestamp"))
                                        vals.append(val)
                                    rows.append(vals)
                                
                                if rows and result["columns"]:
                                    result["rows"] = rows
                                    result["row_count"] = len(rows)
        
        return result

# ---- FastAPI App ----
app = FastAPI(
    title="Genie API Service",
    description="Databricks Genie integration for Microsoft Teams",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Teams bot domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        databricks_host=DATABRICKS_HOST,
        space_id=SPACE_ID
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Alternative health check endpoint"""
    return HealthResponse(
        status="healthy",
        databricks_host=DATABRICKS_HOST,
        space_id=SPACE_ID
    )

@app.post("/genie/query", response_model=QueryResponse)
async def query_genie(request: QueryRequest):
    """
    Main endpoint to query Databricks Genie
    
    - **user_id**: Unique identifier for the user (from Teams)
    - **question**: The question to ask Genie
    - **session_id**: Optional conversation ID to continue existing conversation
    - **max_wait**: Maximum seconds to wait for response (default 120)
    """
    logger.info(f"Query from user {request.user_id}: {request.question}")
    
    try:
        # Get or create conversation for this user
        conversation_id = request.session_id or user_sessions.get(request.user_id)
        
        # Create Genie client
        client = GenieClient(DATABRICKS_HOST, SPACE_ID, HEADERS)
        
        # Start new conversation if needed
        if not conversation_id:
            conversation_id = client.start_conversation()
            user_sessions[request.user_id] = conversation_id
        else:
            client.conversation_id = conversation_id
        
        # Send message and get response
        message_id = client.send_message(request.question)
        message_data = client.get_message_status(message_id, max_wait=request.max_wait)
        
        # Extract response
        result = client.extract_response(message_data, message_id)
        
        # Build response
        answer_text = result.get("analysis", "No analysis provided")
        if not answer_text or answer_text.strip() == "":
            answer_text = "Query completed successfully"
        
        return QueryResponse(
            session_id=conversation_id,
            answer_text=answer_text,
            sql=result.get("sql"),
            columns=result.get("columns"),
            rows=result.get("rows"),
            row_count=result.get("row_count", 0),
            status=result.get("status", "UNKNOWN")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.delete("/genie/session/{user_id}")
async def clear_session(user_id: str):
    """Clear conversation session for a user"""
    if user_id in user_sessions:
        del user_sessions[user_id]
        return {"status": "cleared", "user_id": user_id}
    return {"status": "not_found", "user_id": user_id}

# Add at the end of app.py, before if __name__ == "__main__":

from fastapi.openapi.utils import get_openapi

def custom_openapi_swagger2():
    """Generate Swagger 2.0 for PowerApps"""
    openapi_3 = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Basic Swagger 2.0 structure
    swagger_2 = {
        "swagger": "2.0",
        "info": openapi_3["info"],
        "host": "",
        "basePath": "/",
        "schemes": ["https", "http"],
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "paths": {},
        "definitions": {}
    }
    
    # Convert paths
    for path, methods in openapi_3.get("paths", {}).items():
        swagger_2["paths"][path] = {}
        for method, op in methods.items():
            if method in ["get", "post", "delete"]:
                swagger_op = {
                    "summary": op.get("summary", ""),
                    "description": op.get("description", ""),
                    "operationId": op.get("operationId", ""),
                    "parameters": op.get("parameters", []),
                    "responses": {}
                }
                
                # Convert requestBody to body parameter
                if "requestBody" in op:
                    body_schema = op["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})
                    if "$ref" in body_schema:
                        ref = body_schema["$ref"].replace("#/components/schemas/", "#/definitions/")
                        swagger_op["parameters"].append({
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {"$ref": ref}
                        })
                
                # Convert responses
                for code, resp in op.get("responses", {}).items():
                    resp_schema = resp.get("content", {}).get("application/json", {}).get("schema", {})
                    swagger_resp = {"description": resp.get("description", "Success")}
                    if "$ref" in resp_schema:
                        ref = resp_schema["$ref"].replace("#/components/schemas/", "#/definitions/")
                        swagger_resp["schema"] = {"$ref": ref}
                    swagger_op["responses"][code] = swagger_resp
                
                swagger_2["paths"][path][method] = swagger_op
    
    # Convert schemas to definitions
    for name, schema in openapi_3.get("components", {}).get("schemas", {}).items():
        # Simplify anyOf (PowerApps doesn't like it)
        def simplify(s):
            if isinstance(s, dict):
                if "anyOf" in s:
                    return next((o for o in s["anyOf"] if o.get("type") != "null"), s["anyOf"][0])
                return {k: (simplify(v) if k == "properties" else v) for k, v in s.items()}
            return s
        swagger_2["definitions"][name] = simplify(schema)
    
    return swagger_2

@app.get("/swagger.json", include_in_schema=False)
async def get_swagger_2():
    """Swagger 2.0 definition for PowerApps"""
    return custom_openapi_swagger2()

if __name__ == "__main__":
    import uvicorn
    # Dynamically grab the port from App Runner, default to 8080 for local dev
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
