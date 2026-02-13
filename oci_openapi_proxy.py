import os
import time
import json
import uuid
from typing import List, Optional, Dict, Any, Iterable

import requests
import oci
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

# ============================================================
# CONFIG
# ============================================================

OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config"))
OCI_PROFILE = os.getenv("OCI_PROFILE", "DEFAULT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "<YOUR_COMPARTMENT_ID>")
OCI_GENAI_ENDPOINT = os.getenv(
    "OCI_GENAI_ENDPOINT",
    "https://inference.generativeai.<region>.oci.oraclecloud.com"
)

MODEL_MAP = {
    "gpt-4o-mini": "openai.gpt-4.1",
    "text-embedding-3-small": "cohere.embed-multilingual-v3.0",
}

app = FastAPI(title="OCI OpenAI-Compatible Gateway")

# ============================================================
# Pydantic Models (OpenAI-compatible)
# ============================================================

class Message(BaseModel):
    role: str
    content: str

class EmbeddingRequest(BaseModel):
    model: str
    input: List[str] | str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None

    model_config = ConfigDict(extra="allow")

# ============================================================
# OCI SIGNER
# ============================================================

def get_signer():
    config = oci.config.from_file(OCI_CONFIG_FILE, OCI_PROFILE)
    signer = oci.signer.Signer(
        tenancy=config["tenancy"],
        user=config["user"],
        fingerprint=config["fingerprint"],
        private_key_file_location=config["key_file"],
        pass_phrase=config.get("pass_phrase"),
    )
    return signer


# ============================================================
# CONVERSION HELPERS
# ============================================================

def openai_to_oci_messages(messages: list, model_id: str) -> list:
    oci_messages = []

    for m in messages:
        role = m.get("role", "").upper()

        if role == "SYSTEM":
            role = "SYSTEM"
        elif role == "ASSISTANT":
            role = "ASSISTANT"
        else:
            role = "USER"

        oci_messages.append({
            "role": role,
            "content": [
                {
                    "type": "TEXT",
                    "text": m.get("content", "")
                }
            ]
        })

    return oci_messages

def build_openai_response(model: str, text: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }

def normalize_messages(messages: list) -> list:
    normalized = []

    for m in messages:
        content = m.get("content")

        # Caso OpenClaw envie array [{type:"text", text:"..."}]
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)

        normalized.append({
            "role": m.get("role"),
            "content": content
        })

    return normalized

def fake_stream(text: str, model: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield f"data: {json.dumps({
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {'role': 'assistant'},
            'finish_reason': None
        }]
    })}\n\n"

    for i in range(0, len(text), 40):
        chunk = text[i:i+40]
        yield f"data: {json.dumps({
            'id': completion_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {'content': chunk},
                'finish_reason': None
            }]
        })}\n\n"

    yield "data: [DONE]\n\n"

# ============================================================
# OCI CHAT CALL (REST 20231130)
# ============================================================

def call_oci_chat(request: dict) -> str:
    signer = get_signer()

    model = request.get("model")
    oci_model = MODEL_MAP.get(model, model)

    url = f"{OCI_GENAI_ENDPOINT}/20231130/actions/chat"

    oci_messages = []
    for m in request.get("messages", []):
        oci_messages.append({
            "role": m["role"].upper(),
            "content": [
                {
                    "type": "TEXT",
                    "text": m["content"]
                }
            ]
        })

    payload = {
        "compartmentId": OCI_COMPARTMENT_ID,
        "servingMode": {
            "servingType": "ON_DEMAND",
            "modelId": oci_model
        },
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": oci_messages,
            "maxTokens": request.get("max_tokens", 512),
            "temperature": request.get("temperature", 0.7),
            "topP": request.get("top_p", 0.9)
        }
    }

    print("\n================ OCI PAYLOAD ================")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=============================================\n")

    response = requests.post(
        url,
        json=payload,
        auth=signer,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        print("\n================ OCI ERROR =================")
        print(response.text)
        print("===========================================\n")
        raise HTTPException(status_code=500, detail=response.text)

    data = response.json()

    # Caminho correto da resposta GENERIC
    choices = data["chatResponse"]["choices"]
    message = choices[0]["message"]
    content = message["content"]

    return content[0]["text"]

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "oci"}
            for k in MODEL_MAP.keys()
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    print("\n=== OPENCLAW BODY ===")
    print(json.dumps(body, indent=2))
    print("=====================\n")

    body["messages"] = normalize_messages(body["messages"])
    text = call_oci_chat(body)

    if body.get("stream"):
        return StreamingResponse(
            fake_stream(text, body["model"]),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    return build_openai_response(body["model"], text)

# ============================================================
# HEALTHCHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()

    try:
        body_json = json.loads(body.decode())
    except:
        body_json = body.decode()

    print("\n>>> HIT:", request.method, request.url.path)
    print(">>> BODY:", json.dumps(body_json, indent=2, ensure_ascii=False))

    # NÃO mexe no request._receive
    response = await call_next(request)
    return response


@app.post("/v1/responses")
async def responses_passthrough(request: Request):
    body = await request.json()

    body["messages"] = normalize_messages(body.get("messages", []))
    text = call_oci_chat(body)

    if body.get("stream"):
        return StreamingResponse(
            fake_stream(text, body["model"]),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    return build_openai_response(body["model"], text)

