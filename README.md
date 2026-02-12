# Integrating OpenClaw with Oracle Cloud Generative AI (OCI)

## Overview

This tutorial explains how to integrate **OpenClaw** with **Oracle Cloud
Infrastructure (OCI) Generative AI** by building an OpenAI-compatible
API gateway using FastAPI.

Instead of modifying OpenClaw's core, we expose an **OpenAI-compatible
endpoint** (`/v1/chat/completions`) that internally routes requests to
OCI Generative AI.

This approach provides:

-   ✅ Full OpenClaw compatibility
-   ✅ Control over OCI model mapping
-   ✅ Support for streaming responses
-   ✅ Enterprise-grade OCI infrastructure
-   ✅ Secure request signing via OCI SDK

------------------------------------------------------------------------

# Why Use OCI Generative AI?

Oracle Cloud Infrastructure provides:

-   Enterprise security (IAM, compartments, VCN)
-   Flexible model serving (ON_DEMAND, Dedicated)
-   High scalability
-   Cost control
-   Regional deployment control
-   Native integration with Oracle ecosystem

By building an OpenAI-compatible proxy, we combine:

OpenClaw flexibility + OCI enterprise power

------------------------------------------------------------------------

# Architecture

OpenClaw ↓ OpenAI-Compatible Gateway (FastAPI) ↓ OCI Generative AI REST
API (20231130) ↓ OCI Hosted LLM

------------------------------------------------------------------------

# Project Structure

    project/
     ├── oci_openai_proxy.py
     ├── README.md

------------------------------------------------------------------------

# Key Code Sections Explained

## 1️⃣ Configuration Section

``` python
OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config"))
OCI_PROFILE = os.getenv("OCI_PROFILE", "DEFAULT")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "...")
OCI_GENAI_ENDPOINT = os.getenv(
    "OCI_GENAI_ENDPOINT",
    "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
)
```

### What it does:

-   Reads OCI authentication config
-   Defines target compartment
-   Defines the OCI inference endpoint

------------------------------------------------------------------------

## 2️⃣ Model Mapping

``` python
MODEL_MAP = {
    "gpt-4o-mini": "openai.gpt-4.1",
    "text-embedding-3-small": "cohere.embed-multilingual-v3.0",
}
```

### Why this is important:

OpenClaw expects OpenAI model names.\
OCI uses different model IDs.

This dictionary translates between them.

------------------------------------------------------------------------

## 3️⃣ Pydantic OpenAI-Compatible Request Model

``` python
class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
```

### Purpose:

Defines a request format fully compatible with OpenAI's API.

------------------------------------------------------------------------

## 4️⃣ OCI Signer

``` python
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
```

### Purpose:

Creates a signed request for OCI REST calls.

Without this, OCI rejects the request.

------------------------------------------------------------------------

## 5️⃣ Message Conversion (OpenAI → OCI Format)

``` python
def openai_to_oci_messages(messages: list, model_id: str) -> list:
```

OCI expects:

    {
      "role": "USER",
      "content": [
        {"type": "TEXT", "text": "..."}
      ]
    }

OpenAI sends:

    { "role": "user", "content": "..." }

This function converts formats.

------------------------------------------------------------------------

## 6️⃣ OCI REST Call

``` python
url = f"{OCI_GENAI_ENDPOINT}/20231130/actions/chat"
```

We use OCI's REST endpoint:

    POST /20231130/actions/chat

Payload structure:

    {
      "compartmentId": "...",
      "servingMode": {
        "servingType": "ON_DEMAND",
        "modelId": "openai.gpt-4.1"
      },
      "chatRequest": {
        "apiFormat": "GENERIC",
        "messages": [...],
        "maxTokens": 512
      }
    }

------------------------------------------------------------------------

## 7️⃣ Streaming Implementation

``` python
def fake_stream(text: str, model: str):
```

Since OCI GENERIC mode returns full response (not streaming), we
simulate OpenAI streaming by splitting the response into chunks.

This keeps OpenClaw fully compatible.

------------------------------------------------------------------------

## 8️⃣ OpenAI-Compatible Response Builder

``` python
def build_openai_response(model: str, text: str)
```

Formats the OCI response to match OpenAI's schema:

    {
      "id": "...",
      "object": "chat.completion",
      "choices": [...]
    }

------------------------------------------------------------------------

# Running the Server

Install dependencies:

    pip install fastapi uvicorn requests oci pydantic

Run:

    uvicorn oci_openai_proxy:app --host 0.0.0.0 --port 8050

------------------------------------------------------------------------

# Testing with curl

    curl http://127.0.0.1:8050/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "gpt-4o-mini",
        "messages": [
          {"role": "user", "content": "Hello"}
        ]
      }'

------------------------------------------------------------------------

# OpenClaw Configuration (openclaw.json)

Edit your **openclaw.json** configuration file (normaly it's in ~/.openclaw/openclaw.json) and replace models and agents definitions with:

```json
{
   "models":{
      "providers":{
         "openai-compatible":{
            "baseUrl":"http://127.0.0.1:8050/v1",
            "apiKey":"sk-test",
            "api":"openai-completions",
            "models":[
               {
                  "id":"gpt-4o-mini",
                  "name":"gpt-4o-mini",
                  "reasoning":false,
                  "input":[
                     "text"
                  ],
                  "contextWindow":200000,
                  "maxTokens":8192
               }
            ]
         }
      }
   },
   "agents":{
      "defaults":{
         "model":{
            "primary":"openai-compatible/gpt-4o-mini"
         }
      }
   },
   "gateway":{
      "port":18789,
      "mode":"local",
      "bind":"loopback"
   }
}
```

### Important Fields

```table
  Field           Purpose
  --------------- --------------------------------
  baseUrl         Points OpenClaw to our gateway
  api             Must be openai-completions
  model id        Must match MODEL_MAP key
  contextWindow   Model context size
  maxTokens       Max response tokens
```

------------------------------------------------------------------------

# Final Notes

You now have:

✔ OpenClaw fully integrated\
✔ OCI Generative AI backend\
✔ Streaming compatibility\
✔ Enterprise-ready architecture

------------------------------------------------------------------------

# Reference

- [Installing the OCI CLI](https://docs.oracle.com/en-us/iaas/private-cloud-appliance/pca/installing-the-oci-cli.htm)
- [Oracle Cloud Generative AI](https://www.oracle.com/artificial-intelligence/generative-ai/generative-ai-service/)
- [OpenClaw](https://openclaw.ai/)

# Acknowledgments

- **Author** - Cristiano Hoshikawa (Oracle LAD A-Team Solution Engineer)
