"""
LLM Service — Groq (primary) → OpenRouter (secondary) → Ollama (offline fallback)

Priority:
  1. Groq API         → cloud, extremely fast (~200 tok/sec), free
  2. OpenRouter API   → cloud, fallback if Groq fails
  3. Local Ollama     → offline fallback, uses qwen2.5:7b

Two modes:
  1. Chat / RAG answer   — conversational streaming response with context
  2. Structured extract  — forces JSON output for product extraction
"""

import json
import logging
import re
from typing import Any, AsyncGenerator, Optional, List, Dict

import httpx
import ollama
from ollama import AsyncClient

from config import settings

logger = logging.getLogger(__name__)

# ── Ollama client (lazy) ───────────────────────────────────────
_async_ollama: Optional[AsyncClient] = None

def _get_ollama_client() -> AsyncClient:
    global _async_ollama
    if _async_ollama is None:
        _async_ollama = AsyncClient(host=settings.OLLAMA_BASE_URL)
    return _async_ollama


# ── Provider helpers ───────────────────────────────────────────
def _groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

def _openrouter_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Data Fetcher",
    }

def _use_groq() -> bool:
    return bool(settings.GROQ_API_KEY)

def _use_openrouter() -> bool:
    return bool(settings.OPENROUTER_API_KEY)


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

SYSTEM_CHAT = """You are an intelligent document analysis assistant.
You help users extract information from product brochures and PDF documents.
Always base your answers strictly on the provided document context.
If the answer is not in the context, say so honestly.
When asked to extract data, return it as a structured list."""

SYSTEM_EXTRACT = """You are a precise data extraction engine.
Your ONLY job is to extract product information from the given text.
You MUST return valid JSON — nothing else.
Extract EVERY product or item you find, with ALL available details.
Be thorough and accurate.

CRITICAL FIELD DEFINITIONS:
1. 'name': MUST be included for EVERY product object. It is the high-level product line, category, or section header (e.g., "FINISH & TRIM NAILERS"). If a chunk lists multiple products under the same category, you MUST explicitly repeat that category 'name' for EACH AND EVERY product in the JSON array. Do not use dashes or leave it blank.
2. 'model': The specific descriptive name of the model/tool (e.g., "Cordless 15 GA Angled Finish Nailer"). Do NOT use the alphanumeric SKU or code as the model.
3. 'sku': The alphanumeric part number, SKU code, or catalog identifier (e.g., "GFN1564K", "GBT1850K"). Check columns like "SKU #", "Part #", or "Model #".
"""


# ──────────────────────────────────────────────
# Generic OpenAI-compatible streaming helper
# ──────────────────────────────────────────────

async def _openai_compat_stream(base_url: str, headers: dict, model: str, messages: list) -> AsyncGenerator[str, None]:
    """Stream tokens from any OpenAI-compatible API (Groq, OpenRouter, etc.)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def _openai_compat_complete(base_url: str, headers: dict, model: str, messages: list) -> str:
    """Non-streaming completion from any OpenAI-compatible API."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

async def chat_with_context(
    message: str,
    context_chunks: List[Dict],
    history: Optional[List[Dict]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a RAG-grounded chat response token by token.
    Tries Groq → OpenRouter → Ollama in order.
    """
    context_text = _format_context(context_chunks)
    user_prompt = f"""Use the following document context to answer the question.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION: {message}

Answer based strictly on the context above."""

    messages = [{"role": "system", "content": SYSTEM_CHAT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_prompt})

    # ── 1. Try Groq (primary) ──────────────────
    if _use_groq():
        try:
            logger.info(f"Using Groq ({settings.GROQ_MODEL})")
            async for token in _openai_compat_stream(
                settings.GROQ_BASE_URL, _groq_headers(), settings.GROQ_MODEL, messages
            ):
                yield token
            return
        except Exception as exc:
            logger.warning(f"Groq failed ({exc}), trying OpenRouter...")

    # ── 2. Try OpenRouter (secondary) ─────────
    if _use_openrouter():
        try:
            logger.info(f"Using OpenRouter ({settings.OPENROUTER_MODEL})")
            async for token in _openai_compat_stream(
                settings.OPENROUTER_BASE_URL, _openrouter_headers(), settings.OPENROUTER_MODEL, messages
            ):
                yield token
            return
        except Exception as exc:
            logger.warning(f"OpenRouter failed ({exc}), falling back to Ollama...")

    # ── 3. Fallback: Local Ollama ──────────────
    logger.info("Using local Ollama (offline fallback)")
    async for chunk in await _get_ollama_client().chat(
        model=settings.OLLAMA_LLM_MODEL,
        messages=messages,
        stream=True,
        keep_alive=-1,
        options={"num_ctx": 2048},
    ):
        token = chunk["message"]["content"]
        if token:
            yield token


async def _process_extraction_batch(batch: List[Dict], batch_index: int, total_batches: int, state: Optional[dict] = None) -> List[Dict]:
    """Helper to process a single batch with fallback logic."""
    context_text = _format_context(batch)
    user_prompt = f"""Extract all product information from this document text.
Return a JSON array where each element is a product with all its details.

For EVERY product object, you MUST extract:
- 'name': The high-level product line, category, or section header of the page/block (e.g., "FINISH & TRIM NAILERS", "CORDLESS NAILERS"). If multiple products share the same category, explicitly REPEAT the category name for EACH product. Do not leave it blank or use dashes.
- 'model': The specific descriptive name of the model/tool (e.g., "Cordless 15 GA Angled Finish Nailer"). Never use the SKU code or alphanumeric model number as the model name.
- 'sku': The alphanumeric part number, SKU code, or catalog identifier (e.g., "GFN1564K"). Look for it in specifications or column headers (like "SKU #", "Part #", or "Model #").

Include any other available details as fields, such as: price, specifications, features, dimensions, weight, capacity, operating_pressure, warranty, description.

DOCUMENT TEXT:
{context_text}

Return ONLY the JSON array, no explanation."""

    messages = [
        {"role": "system", "content": SYSTEM_EXTRACT},
        {"role": "user", "content": user_prompt},
    ]

    raw = None

    # ── 1. Try Groq ────────────────────────
    if _use_groq() and not (state and state.get("groq_disabled", False)):
        retries = 3
        for attempt in range(retries):
            try:
                logger.info(f"Groq extraction batch {batch_index}/{total_batches} (Attempt {attempt+1})")
                raw = await _openai_compat_complete(
                    settings.GROQ_BASE_URL, _groq_headers(), settings.GROQ_MODEL, messages
                )
                break
            except Exception as exc:
                if attempt < retries - 1:
                    sleep_time = 61 if "429" in str(exc) else 15
                    logger.warning(f"Groq attempt {attempt+1} failed ({exc}). Retrying in {sleep_time} seconds...")
                    import asyncio
                    await asyncio.sleep(sleep_time)
                else:
                    if state is not None:
                        state["groq_disabled"] = True
                    logger.warning(f"Groq extraction failed for batch {batch_index} ({exc}). Switching to OpenRouter fallback for remaining batches.")

    # ── 2. Try OpenRouter ──────────────────
    if raw is None and _use_openrouter() and not (state and state.get("openrouter_disabled", False)):
        retries = 3
        for attempt in range(retries):
            try:
                logger.info(f"OpenRouter extraction batch {batch_index}/{total_batches} (Attempt {attempt+1})")
                raw = await _openai_compat_complete(
                    settings.OPENROUTER_BASE_URL, _openrouter_headers(), settings.OPENROUTER_MODEL, messages
                )
                break
            except Exception as exc:
                if attempt < retries - 1:
                    logger.warning(f"OpenRouter attempt {attempt+1} failed ({exc}). Retrying in 5 seconds...")
                    import asyncio
                    await asyncio.sleep(5)
                else:
                    if state is not None:
                        state["openrouter_disabled"] = True
                    logger.warning(f"OpenRouter extraction failed for batch {batch_index} ({exc}). Switching to local Ollama fallback for remaining batches.")

    # ── 3. Fallback: Ollama ────────────────
    if raw is None:
        try:
            logger.info(f"Ollama extraction batch {batch_index}/{total_batches} (offline fallback)")
            response = await _get_ollama_client().chat(
                model=settings.OLLAMA_LLM_MODEL,
                messages=messages,
                format="json",
                keep_alive=-1,
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            raw = response["message"]["content"]
        except Exception as exc:
            logger.error(f"All providers failed for batch {batch_index}: {exc}")
            return []

    return _parse_json_products(raw)

import asyncio

async def extract_products(
    context_chunks: List[Dict],
    on_progress: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Extract ALL product details from context as structured JSON.
    Scans batches sequentially with pacing to respect rate limits.
    """
    all_products = []
    batch_size = 6  # Small enough to stay under TPM limit per request
    batches = [context_chunks[i:i + batch_size] for i in range(0, len(context_chunks), batch_size)]
    total_batches = len(batches)
    state = {
        "groq_disabled": False,
        "openrouter_disabled": False
    }

    for idx, batch in enumerate(batches):
        batch_index = idx + 1
        products = await _process_extraction_batch(batch, batch_index, total_batches, state)
        if products:
            all_products.extend(products)
        
        if on_progress:
            try:
                if asyncio.iscoroutinefunction(on_progress):
                    await on_progress(batch_index, total_batches, products)
                else:
                    on_progress(batch_index, total_batches, products)
            except Exception as e:
                logger.error(f"Error calling progress callback: {e}")
                
        # Pace requests with a 25-second sleep between batches to preserve rate limit window
        if batch_index < total_batches:
            # If both cloud providers are disabled (local Ollama fallback active), run at max speed without sleep
            if state.get("groq_disabled", False) and state.get("openrouter_disabled", False):
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(25)

    # Forward-fill missing 'name' fields if any products are missing them, using the last known name.
    last_known_name = ""
    for p in all_products:
        name = str(p.get("name", "")).strip()
        if name and name not in ("—", "-", "None", "null", ""):
            last_known_name = name
        elif last_known_name:
            p["name"] = last_known_name

    return all_products


async def check_ollama_status() -> dict:
    """Check provider status and model availability."""
    result = {
        "groq": {
            "configured": _use_groq(),
            "model": settings.GROQ_MODEL,
        },
        "openrouter": {
            "configured": _use_openrouter(),
            "model": settings.OPENROUTER_MODEL,
        },
    }
    try:
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        models_response = client.list()
        available = [m["name"] for m in models_response.get("models", [])]
        result["running"] = True
        result["available_models"] = available
        result["llm_ready"] = any(settings.OLLAMA_LLM_MODEL in m for m in available)
        result["embed_ready"] = any(settings.OLLAMA_EMBED_MODEL in m for m in available)
    except Exception as exc:
        result["running"] = False
        result["error"] = str(exc)
    return result


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk.get('filename', 'doc')}, Page {chunk.get('page', '?')}]"
        parts.append(f"--- Source {i}: {source} ---\n{chunk['text']}")
    return "\n\n".join(parts)


def _parse_json_products(raw: str) -> List[Dict]:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        for key in ("products", "items", "data", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse failed: {exc}. Trying robust partial parser...")
        
        # Recover fully formed JSON objects from a truncated array
        products = []
        start_idx = 0
        while True:
            start_pos = raw.find('{', start_idx)
            if start_pos == -1:
                break
            
            brace_count = 0
            end_pos = -1
            in_string = False
            escape = False
            for i in range(start_pos, len(raw)):
                char = raw[i]
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
            
            if end_pos != -1:
                obj_str = raw[start_pos:end_pos+1]
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and any(k in obj for k in ("name", "model", "title")):
                        products.append(obj)
                except Exception:
                    pass
                start_idx = end_pos + 1
            else:
                break
                
        if products:
            logger.info(f"Successfully recovered {len(products)} products from truncated JSON response.")
            return products

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return []
