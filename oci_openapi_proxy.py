import os
import time
import json
import uuid
from typing import Optional, List, Dict, Any
import re
import subprocess

import requests
import oci
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
import requests
import os

import requests

def get_weather_from_api(city: str) -> str:
    """
    Consulta clima atual usando Open-Meteo (100% free, sem API key)
    """
    print("LOG: EXECUTE TOOL WEATHER")
    try:
        # 1️⃣ Geocoding (cidade -> lat/lon)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "pt",
            "format": "json"
        }

        geo_response = requests.get(geo_url, params=geo_params, timeout=10)

        if geo_response.status_code != 200:
            return f"Erro geocoding: {geo_response.text}"

        geo_data = geo_response.json()

        if "results" not in geo_data or len(geo_data["results"]) == 0:
            return f"Cidade '{city}' não encontrada."

        location = geo_data["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        resolved_name = location["name"]
        country = location.get("country", "")

        # 2️⃣ Clima atual
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True,
            "timezone": "auto"
        }

        weather_response = requests.get(weather_url, params=weather_params, timeout=10)

        if weather_response.status_code != 200:
            return f"Erro clima: {weather_response.text}"

        weather_data = weather_response.json()

        current = weather_data.get("current_weather")

        if not current:
            return "Dados de clima indisponíveis."

        temperature = current["temperature"]
        windspeed = current["windspeed"]

        return (
            f"Temperatura atual em {resolved_name}, {country}: {temperature}°C.\n"
            f"Velocidade do vento: {windspeed} km/h."
        )

    except Exception as e:
        return f"Weather tool error: {str(e)}"
# ============================================================
# CONFIG
# ============================================================

OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config"))
OCI_PROFILE = os.getenv("OCI_PROFILE", "DEFAULT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..aaaaaaaaexpiw4a7dio64mkfv2t273s2hgdl6mgfvvyv7tycalnjlvpvfl3q")
OCI_GENAI_ENDPOINT = os.getenv(
    "OCI_GENAI_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
)

OPENCLAW_TOOLS_ACTIVE = True

SYSTEM_AGENT_PROMPT = """
You are an enterprise AI agent.

You MUST respond ONLY in valid JSON.

Available tools:
- weather(city: string)

Response format:

If you need to call a tool:
{
  "action": "call_tool",
  "tool": "<tool_name>",
  "arguments": { ... }
}

If you are returning a final answer:
{
  "action": "final_answer",
  "content": "<final user answer>"
}

Never include explanations outside JSON.
"""

TOOLS = {
    "weather": lambda city: get_weather_from_api(city)
}

if not OCI_COMPARTMENT_ID:
    raise RuntimeError("OCI_COMPARTMENT_ID not defined")

# Mapeamento OpenAI → OCI
MODEL_MAP = {
    "gpt-5": "openai.gpt-4.1",
    "openai/gpt-5": "openai.gpt-4.1",
    "openai-compatible/gpt-5": "openai.gpt-4.1",
}

app = FastAPI(title="OCI OpenAI-Compatible Gateway")

# ============================================================
# OCI SIGNER
# ============================================================

def get_signer():
    config = oci.config.from_file(OCI_CONFIG_FILE, OCI_PROFILE)
    return oci.signer.Signer(
        tenancy=config["tenancy"],
        user=config["user"],
        fingerprint=config["fingerprint"],
        private_key_file_location=config["key_file"],
        pass_phrase=config.get("pass_phrase"),
    )

# ============================================================
# OCI CHAT CALL (OPENAI FORMAT)
# ============================================================

def _openai_messages_to_generic(messages: list) -> list:
    """
    OpenAI:  {"role":"user","content":"..."}
    Generic: {"role":"USER","content":[{"type":"TEXT","text":"..."}]}
    """
    out = []
    for m in messages or []:
        role = (m.get("role") or "user").upper()

        # OCI GENERIC geralmente espera USER/ASSISTANT
        if role == "SYSTEM":
            role = "USER"
        elif role == "TOOL":
            role = "USER"

        content = m.get("content", "")

        # Se vier lista (OpenAI multimodal), extrai texto
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("text", "TEXT"):
                    parts.append(item.get("text", ""))
            content = "\n".join(parts)

        out.append({
            "role": role,
            "content": [{"type": "TEXT", "text": str(content)}]
        })
    return out


def call_oci_chat(body: dict):
    signer = get_signer()

    model = body.get("model")
    oci_model = MODEL_MAP.get(model, model)

    url = f"{OCI_GENAI_ENDPOINT}/20231130/actions/chat"

    generic_messages = _openai_messages_to_generic(body.get("messages", []))

    payload = {
        "compartmentId": OCI_COMPARTMENT_ID,
        "servingMode": {
            "servingType": "ON_DEMAND",
            "modelId": oci_model
        },
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": generic_messages,
            "maxTokens": int(body.get("max_tokens", 1024)),
            "temperature": float(body.get("temperature", 0.0)),
            "topP": float(body.get("top_p", 1.0)),
        }
    }

    # ⚠️ IMPORTANTÍSSIMO:
    # Em GENERIC, NÃO envie tools/tool_choice/stream (você orquestra tools no proxy)
    # Se você mandar, pode dar 400 "correct format of request".

    print("\n=== PAYLOAD FINAL (GENERIC) ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    r = requests.post(url, json=payload, auth=signer)
    if r.status_code != 200:
        print("OCI ERROR:", r.text)
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()["chatResponse"]

def detect_tool_call(text: str):
    pattern = r"exec\s*\(\s*([^\s]+)\s*(.*?)\s*\)"
    match = re.search(pattern, text)

    if not match:
        return None

    tool_name = "exec"
    command = match.group(1)
    args = match.group(2)

    return {
        "tool": tool_name,
        "args_raw": f"{command} {args}".strip()
    }

def execute_exec_command(command: str):
    try:
        print(f"LOG: EXEC COMMAND: {command}")
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        return result.decode()
    except subprocess.CalledProcessError as e:
        return e.output.decode()

def execute_real_tool(name, args):

    if name == "weather":
        city = args.get("city")
        return get_weather_from_api(city)

    return "Tool not implemented"

def _extract_generic_text(oci_message: dict) -> str:
    content = oci_message.get("content")
    if isinstance(content, list):
        return "".join([i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "TEXT"])
    if isinstance(content, str):
        return content
    return str(content)


def agent_loop(body: dict, max_iterations=5):

    # Trabalhe sempre com OpenAI messages internamente,
    # mas call_oci_chat converte pra GENERIC.
    messages = []
    messages.append({"role": "system", "content": SYSTEM_AGENT_PROMPT})
    messages.extend(body.get("messages", []))

    for _ in range(max_iterations):

        response = call_oci_chat({**body, "messages": messages})

        oci_choice = response["choices"][0]
        oci_message = oci_choice["message"]

        text = _extract_generic_text(oci_message)

        try:
            agent_output = json.loads(text)
        except:
            # modelo não retornou JSON (quebrou regra)
            return response

        if agent_output.get("action") == "call_tool":
            tool_name = agent_output.get("tool")
            args = agent_output.get("arguments", {})

            if tool_name not in TOOLS:
                # devolve pro modelo como erro
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": json.dumps({
                    "tool_error": f"Tool '{tool_name}' not implemented"
                })})
                continue

            tool_result = TOOLS[tool_name](**args)

            # Mantém o histórico: (1) decisão do agente, (2) resultado do tool
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": json.dumps({
                "tool_result": {
                    "tool": tool_name,
                    "arguments": args,
                    "result": tool_result
                }
            }, ensure_ascii=False)})

            continue

        if agent_output.get("action") == "final_answer":
            return response

    return response

# ============================================================
# STREAMING ADAPTER
# ============================================================

def stream_openai_format(chat_response: dict, model: str):

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    content = chat_response["choices"][0]["message"]["content"]

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

    for i in range(0, len(content), 60):
        chunk = content[i:i+60]
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
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "oci"}
            for k in MODEL_MAP.keys()
        ],
    }

# ------------------------------------------------------------
# CHAT COMPLETIONS
# ------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    body = await request.json()
    # chat_response = call_oci_chat(body)
    # chat_response = agent_loop(body)

    if OPENCLAW_TOOLS_ACTIVE:
        chat_response = call_oci_chat(body)

        oci_choice = chat_response["choices"][0]
        oci_message = oci_choice["message"]

        content_text = _extract_generic_text(oci_message)

        # 🔥 DETECT EXEC
        exec_match = re.search(r"\(exec\s+(.*?)\)", content_text)

        if exec_match:
            command = exec_match.group(1)
            result = execute_exec_command(command)

            # Injeta resultado e chama novamente
            new_messages = body["messages"] + [
                {"role": "assistant", "content": content_text},
                {"role": "user", "content": f"Tool result:\n{result}"}
            ]

            chat_response = call_oci_chat({
                **body,
                "messages": new_messages
            })
    else:
        # 🔥 Modo enterprise → seu agent_loop controla tools
        chat_response = agent_loop(body)

    print("FINAL RESPONSE:", json.dumps(chat_response, indent=2))

    oci_choice = chat_response["choices"][0]
    oci_message = oci_choice["message"]

    # 🔥 SE É TOOL CALL → RETORNA DIRETO
    if oci_message.get("tool_calls"):
        return chat_response

    content_text = ""

    content = oci_message.get("content")

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "TEXT":
                content_text += item.get("text", "")
    elif isinstance(content, str):
        content_text = content
    else:
        content_text = str(content)

    finish_reason = oci_choice.get("finishReason", "stop")

    # 🔥 SE STREAMING
    if body.get("stream"):
        async def event_stream():
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            # role chunk
            yield f"data: {json.dumps({
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': body['model'],
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant'},
                    'finish_reason': None
                }]
            })}\n\n"

            # content chunks
            for i in range(0, len(content_text), 50):
                chunk = content_text[i:i+50]

                yield f"data: {json.dumps({
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': body['model'],
                    'choices': [{
                        'index': 0,
                        'delta': {'content': chunk},
                        'finish_reason': None
                    }]
                })}\n\n"

            # final chunk
            yield f"data: {json.dumps({
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': body['model'],
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': finish_reason
                }]
            })}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    # 🔥 SE NÃO FOR STREAM
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body["model"],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content_text
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
# ------------------------------------------------------------
# RESPONSES (OpenAI 2024 format)
# ------------------------------------------------------------

@app.post("/v1/responses")
async def responses(request: Request):

    body = await request.json()

    # chat_response = call_oci_chat(body)
    chat_response = agent_loop(body)

    oci_choice = chat_response["choices"][0]
    oci_message = oci_choice["message"]

    content_text = ""

    content = oci_message.get("content")

    if isinstance(content, list):
        for item in content:
            if item.get("type") == "TEXT":
                content_text += item.get("text", "")
    elif isinstance(content, str):
        content_text = content

    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created": int(time.time()),
        "model": body.get("model"),
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content_text
                    }
                ]
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print("\n>>> ENDPOINT:", request.method, request.url.path)

    body = await request.body()
    try:
        body_json = json.loads(body.decode())
        print(">>> BODY:", json.dumps(body_json, indent=2))
    except:
        print(">>> BODY RAW:", body.decode())

    response = await call_next(request)
    print(">>> STATUS:", response.status_code)
    return response