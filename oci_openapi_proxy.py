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

# ============================================================
# CONFIG
# ============================================================

OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config"))
OCI_PROFILE = os.getenv("OCI_PROFILE", "DEFAULT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "<YOUR_OCI_COMPARTMENT_ID>")
OCI_GENAI_ENDPOINT = os.getenv(
    "OCI_GENAI_ENDPOINT",
    "https://inference.generativeai.<oci_region>.oci.oraclecloud.com"
)
if not OCI_COMPARTMENT_ID:
    raise RuntimeError("OCI_COMPARTMENT_ID not defined")

OPENCLAW_TOOLS_ACTIVE = True

# ============================================================
# PROMPTS to adapt for OCI
# ============================================================

SYSTEM_AGENT_PROMPT = """
You are an autonomous software agent.

You have full access to the local machine.

Available tools:
- weather(city: string)
- exec(command: string)

If a system command is required, respond ONLY with:

{
  "action": "call_tool",
  "tool": "exec",
  "arguments": {
    "command": "<shell command>"
  }
}

***VERY IMPORTANT***: A TASK IS CONSIDERED COMPLETED WHEN IT RESULTS IN A ARTIFACT ASKED FROM THE USER

If task is completed:

{
  "action": "final_answer",
  "content": "<result>"
}
"""

RUNNER_PROMPT = (
    "You are a Linux execution agent.\n"
    "\n"
    "OUTPUT CONTRACT (MANDATORY):\n"
    "- You must output EXACTLY ONE of the following per response:\n"
    "  A) (exec <command>)\n"
    "  B) (done <final answer>)\n"
    "\n"
    "STRICT RULES:\n"
    "1) NEVER output raw commands without (exec <command>). Raw commands will be ignored.\n"
    "2) NEVER output explanations, markdown, code fences, bullets, or extra text.\n"
    "3) If you need to create multi-line files, you MUST use heredoc inside (exec <command>), e.g.:\n"
    "   (exec cat > file.py << 'EOF'\n"
    "   ...\n"
    "   EOF)\n"
    "4) If the previous tool result shows an error, your NEXT response must be (exec <command>) to fix it.\n"
    "5) When the artifact is created successfully, end with (done ...).\n"
    "\n"
    "REMINDER: Your response must be only a single parenthesized block."
)

# Mapeamento OpenAI → OCI
MODEL_MAP = {
    "gpt-5": "openai.gpt-4.1",
    "openai/gpt-5": "openai.gpt-4.1",
    "openai-compatible/gpt-5": "openai.gpt-4.1",
}

# ============================================================
# FASTAPI APP
# ============================================================

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

def build_generic_messages(openai_messages: list, system_prompt: str) -> list:
    out = []
    # 1) Injeta o system como PRIMEIRA mensagem USER, com prefixo fixo
    out.append({
        "role": "USER",
        "content": [{"type":"TEXT","text": "SYSTEM:\n" + system_prompt.strip()}]
    })

    # 2) Depois converte o resto, ignorando systems originais
    for m in openai_messages or []:
        role = (m.get("role") or "user").lower()
        if role == "system":
            continue

        r = "USER" if role in ("user", "tool") else "ASSISTANT"
        content = m.get("content", "")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("text","TEXT"):
                    parts.append(item.get("text",""))
            content = "\n".join(parts)

        out.append({
            "role": r,
            "content": [{"type":"TEXT","text": str(content)}]
        })

    return out


def call_oci_chat(body: dict, system_prompt: str):
    signer = get_signer()

    model = body.get("model")
    oci_model = MODEL_MAP.get(model, model)

    url = f"{OCI_GENAI_ENDPOINT}/20231130/actions/chat"

    # generic_messages = _openai_messages_to_generic(body.get("messages", []))
    generic_messages = build_generic_messages(body.get("messages", []), system_prompt)

    payload = {
        "compartmentId": OCI_COMPARTMENT_ID,
        "servingMode": {
            "servingType": "ON_DEMAND",
            "modelId": oci_model
        },
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": generic_messages,
            "maxTokens": int(body.get("max_tokens", 4000)),
            "temperature": float(body.get("temperature", 0.0)),
            "topP": float(body.get("top_p", 1.0)),
        }
    }

    # ⚠️ IMPORTANTÍSSIMO:
    # Em GENERIC, NÃO envie tools/tool_choice/stream (você orquestra tools no proxy)
    # Se você mandar, pode dar 400 "correct format of request".

    # print("\n=== PAYLOAD FINAL (GENERIC) ===")
    # print(json.dumps(payload, indent=2, ensure_ascii=False))

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
        p = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120  # ajuste
        )
        out = (p.stdout or "") + (p.stderr or "")
        return out if out.strip() else f"(no output) exit={p.returncode}"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"

TOOLS = {
    "weather": lambda city: get_weather_from_api(city),
    "exec": lambda command: execute_exec_command(command)
}

def execute_real_tool(name, args):

    if name == "weather":
        city = args.get("city")
        return get_weather_from_api(city)

    return "Tool not implemented"

def _extract_generic_text(oci_message: dict) -> str:
    content = oci_message.get("content")
    if isinstance(content, list):
        r = "".join([i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "TEXT"])
        # print("r", r)
        return r
    if isinstance(content, str):
        # print("content", content)
        return content
    return str(content)


def agent_loop(body: dict, max_iterations=10000):

    # Trabalhe sempre com OpenAI messages internamente,
    # mas call_oci_chat converte pra GENERIC.
    messages = []
    messages.append({"role": "system", "content": SYSTEM_AGENT_PROMPT})
    messages.extend(body.get("messages", []))

    for _ in range(max_iterations):

        response = call_oci_chat({**body, "messages": messages}, SYSTEM_AGENT_PROMPT)

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

EXEC_RE = re.compile(r"\(exec\s+(.+?)\)\s*$", re.DOTALL)
DONE_RE = re.compile(r"\(done\s+(.+?)\)\s*$", re.MULTILINE)

def run_exec_loop(body: dict, max_steps: int = 10000) -> dict:
    # Histórico OpenAI-style
    messages = [{"role":"system"}]
    messages.extend(body.get("messages", []))

    last = None

    for _ in range(max_steps):
        last = call_oci_chat({**body, "messages": messages}, RUNNER_PROMPT)
        print('LLM Result', last)
        msg = last["choices"][0]["message"]
        text = _extract_generic_text(msg) or ""

        m_done = DONE_RE.search(text)
        print("DONE_RE", text)
        print("m_done", m_done)
        if m_done:
            final_text = m_done.group(1).strip()

            # devolve em formato OpenAI no fim
            return {
                **last,
                "choices": [{
                    **last["choices"][0],
                    "message": {"role":"assistant","content": final_text},
                    "finishReason": "stop"
                }]
            }

        m_exec = EXEC_RE.search(text)
        if m_exec:
            command = m_exec.group(1).strip()
            result = execute_exec_command(command)

            messages.append({"role":"assistant","content": text})
            messages.append({"role":"user","content": f"Tool result:\n{result}"})
            continue

        # Se o modelo quebrou o protocolo:
        messages.append({"role":"assistant","content": text})
        messages.append({"role":"user","content": (
            "Protocol error. You MUST reply ONLY with (exec <command>) or (done <final answer>)."
        )})
        continue

    # estourou steps: devolve última resposta (melhor do que travar)
    return last

def verify_task_completion(original_task: str, assistant_output: str) -> bool:
    """
    Retorna True se tarefa estiver concluída.
    Retorna False se ainda precisar continuar.
    """

    verifier_prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict task completion validator.\n"
                "Answer ONLY with DONE or CONTINUE.\n"
                "DONE = the task is fully completed.\n"
                "CONTINUE = more steps are required.\n"
            ),
        },
        {
            "role": "user",
            "content": f"""
Original task:
{original_task}

Last assistant output:
{assistant_output}

Is the task fully completed?
"""
        }
    ]

    response = call_oci_chat({
        "model": "openai-compatible/gpt-5",
        "messages": verifier_prompt,
        "temperature": 0
    }, verifier_prompt[0]["content"])

    text = _extract_generic_text(response["choices"][0]["message"]).strip().upper()

    return text == "DONE"

# ============================================================
# ENTERPRISE TOOLS
# Set the OPENCLAW_TOOLS_ACTIVE = True to automatize OpenClaw execution Tools
# Set the OPENCLAW_TOOLS_ACTIVE = False and implement your own Tools
# ============================================================

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
        chat_response = run_exec_loop(body, max_steps=10000)
    else:
        # 🔥 Modo enterprise → seu agent_loop controla tools
        chat_response = agent_loop(body)

    # print("FINAL RESPONSE:", json.dumps(chat_response, indent=2))

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
    # print("\n>>> ENDPOINT:", request.method, request.url.path)

    body = await request.body()
    try:
        body_json = json.loads(body.decode())
        # print(">>> BODY:", json.dumps(body_json, indent=2))
    except:
        print(">>> BODY RAW:", body.decode())

    response = await call_next(request)
    # print(">>> STATUS:", response.status_code)
    return response