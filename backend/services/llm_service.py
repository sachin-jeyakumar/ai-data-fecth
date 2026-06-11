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
    return bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())

def _use_openrouter() -> bool:
    return bool(settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY.strip())


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

SYSTEM_CHAT = """You are an intelligent document analysis assistant.
You help users extract information from product brochures and PDF documents.
Always base your answers strictly on the provided document context.
If the answer is not in the context, say so honestly.
When asked to extract data, return it as a structured list."""

SYSTEM_EXTRACT = """You are a precise data extraction engine for a RAG-based document parser.
Your ONLY job is to extract product information STRICTLY from the given page text and text tables.
You MUST return valid JSON — nothing else.
Extract EVERY product or item you find, with ALL available details.
Be thorough and accurate. Do NOT output duplicate products.

CRITICAL RAG RULE:
You are strictly forbidden from using outside knowledge. You must ONLY use the exact text provided in the document context. If a value is not explicitly present in the provided text, omit it entirely. DO NOT hallucinate, guess, invent, or bring in outside information.
No permutations or combinations: Do NOT mix features or specifications across different products to create synthetic items. Only output products and specifications exactly as they are associated in the text.

CRITICAL FIELD DEFINITIONS:
1. 'category': The high-level product line or section header (e.g., "FINISH & TRIM NAILERS"). If a chunk lists multiple products under the same category, explicitly repeat that category for EACH AND EVERY product in the JSON array.
2. 'name': The specific descriptive name of the tool/product (e.g., "Cordless 15 GA Angled Finish Nailer").
3. 'model': The alphanumeric part number, SKU code, or catalog identifier (e.g., "GFN1564K"). Look for it in specifications or column headers (like "SKU #", "Part #", or "Model #").
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
        try:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error {exc.response.status_code} from {base_url}: {exc.response.text}")
            raise


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
            logger.warning(f"Groq failed ({exc})")
            if not _use_openrouter():
                raise RuntimeError("Groq API failed, and no other API provider is configured.") from exc

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
            logger.error(f"OpenRouter failed ({exc})")
            raise RuntimeError("OpenRouter API failed, and no other API provider is configured.") from exc

    raise RuntimeError("No active API providers configured (Groq and OpenRouter keys are missing).")


async def _process_table_extraction_batch(page_data: Dict, batch_index: int, total_batches: int, state: Optional[dict] = None) -> List[Dict]:
    """Helper to process a single page's tables with fallback logic."""
    page_num = page_data.get("page", "?")
    page_text = page_data.get("text", "")
    tables = page_data.get("tables", [])
    
    table_strings = []
    for i, table in enumerate(tables):
        table_strings.append(f"--- Table {i+1} ---")
        for row in table:
            table_strings.append(" | ".join(row))
    tables_str = "\n".join(table_strings)

    user_prompt = f"""Extract all product information from the provided page text and structured tables below.
Return a JSON array where each element is a product with all its details.

For EVERY product object, extract the following core fields IF AND ONLY IF they exist in the text:
- 'category': The high-level product line or section header (e.g., "FINISH & TRIM NAILERS"). If it's not explicitly in the page text or table, leave it blank. DO NOT GUESS.
- 'name': The specific descriptive name of the tool/product (e.g., "Cordless 15 GA Angled Finish Nailer"). Do NOT use the SKU here.
- 'model': The alphanumeric part number, SKU code, or catalog identifier (e.g., "GFN1564K"). Look for it in specifications or column headers (like "SKU #", "Part #", or "Model #"). If it's not present, leave it blank. DO NOT guess or infer a model number.

Include any other available details as fields IF AND ONLY IF they exist in the text or table (e.g., specifications, features, dimensions, weight, capacity).

CRITICAL ANTI-HALLUCINATION RULES:
1. DO NOT invent, guess, or assume ANY values, including category and model. If it's not written in the text, it does not exist. Leave it blank or omit it entirely.
2. If a detail (like price, warranty, or dimensions) is NOT explicitly written in the text or table, you MUST omit the field completely. DO NOT make up fake prices or specs.
3. Map every distinct product you find into a product object.
4. DO NOT OUTPUT DUPLICATE ROWS. If multiple rows have exactly the same product data, only include it once.
5. PRECISE DATA MATCHING: Ensure the extracted text matches the PDF exactly, without typos or modifications.
6. ABSOLUTELY NO PERMUTATIONS OR COMBINATIONS. Do NOT mix and match features to create fake products. If the text says "15 GA Angled" and "16 GA Straight", DO NOT invent "15 GA Straight". Output ONLY the exact lines that physically exist in the text.

PAGE TEXT CONTEXT (Contains products if tables are empty):
{page_text}

STRUCTURED TABLES (If available):
{tables_str}

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
                exc_str = ""
                if hasattr(exc, "response") and exc.response is not None:
                    exc_str = exc.response.text.lower()
                else:
                    exc_str = str(exc).lower()

                if "tpd" in exc_str or "tokens per day" in exc_str or "daily" in exc_str or "rate limit reached" in exc_str:
                    logger.warning(f"Groq daily token limit exceeded. Disabling Groq and falling back immediately. ({exc})")
                    if state is not None:
                        state["groq_disabled"] = True
                    break

                if attempt < retries - 1:
                    sleep_time = 61 if ("429" in exc_str or "429" in str(exc).lower()) else 15
                    logger.warning(f"Groq attempt {attempt+1} failed ({exc}). Retrying in {sleep_time} seconds...")
                    import asyncio
                    await asyncio.sleep(sleep_time)
                else:
                    if state is not None:
                        state["groq_disabled"] = True
                    logger.warning(f"Groq extraction failed for batch {batch_index} ({exc}). Switching to OpenRouter fallback for remaining batches.")

    # ── 2. Try OpenRouter ──────────────────
    if raw is None and _use_openrouter() and not (state and state.get("openrouter_disabled", False)):
        # Determine fallback models to try
        default_model = settings.OPENROUTER_MODEL
        models_to_try = [default_model]
        for fallback_model in ["google/gemma-4-31b-it:free", "qwen/qwen3-coder:free", "meta-llama/llama-3.2-3b-instruct:free", "nousresearch/hermes-3-llama-3.1-405b:free"]:
            if fallback_model != default_model:
                models_to_try.append(fallback_model)
                
        # If we identified a working model in a previous batch, try it first
        current_working_model = state.get("openrouter_model") if state else None
        if current_working_model and current_working_model in models_to_try:
            models_to_try.remove(current_working_model)
            models_to_try.insert(0, current_working_model)

        success = False
        for model in models_to_try:
            if success:
                break
            retries = 2  # Keep retries low per model to speed up fallback
            for attempt in range(retries):
                try:
                    logger.info(f"OpenRouter extraction batch {batch_index}/{total_batches} using {model} (Attempt {attempt+1})")
                    raw = await _openai_compat_complete(
                        settings.OPENROUTER_BASE_URL, _openrouter_headers(), model, messages
                    )
                    success = True
                    if state:
                        state["openrouter_model"] = model
                    break
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if attempt < retries - 1:
                        sleep_time = 10 if "429" in exc_str else 5
                        logger.warning(f"OpenRouter model {model} attempt {attempt+1} failed ({exc}). Retrying in {sleep_time}s...")
                        import asyncio
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.warning(f"OpenRouter model {model} failed all attempts. Trying next fallback model...")

        if not success:
            if state is not None:
                state["openrouter_disabled"] = True
            logger.warning(f"OpenRouter extraction failed for batch {batch_index}. All OpenRouter fallback models failed.")

    # ── 3. Fallback: Ollama (Disabled by user request) ────────────────
    if raw is None:
        logger.error(f"All configured API providers failed for batch {batch_index}")
        return []

    return _parse_json_products(raw)

import asyncio

async def extract_products_from_tables(
    pdf_pages_data: List[Dict],
    on_progress: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Extract ALL product details from pdfplumber structured tables as JSON.
    Scans pages sequentially with pacing to respect rate limits.
    """
    all_products = []
    total_batches = len(pdf_pages_data)
    state = {
        "groq_disabled": False,
        "openrouter_disabled": False,
        "openrouter_model": settings.OPENROUTER_MODEL
    }

    for idx, page_data in enumerate(pdf_pages_data):
        batch_index = idx + 1
        products = await _process_table_extraction_batch(page_data, batch_index, total_batches, state)
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
                
        # Pace requests with a 12-second sleep between batches to preserve rate limit window
        if batch_index < total_batches:
            await asyncio.sleep(12)


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
