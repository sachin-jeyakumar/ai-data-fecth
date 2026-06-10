import asyncio
from services import llm_service

async def main():
    chunks = [{"filename": "test.pdf", "page": 1, "text": "The new SmartWatch Pro 5 costs $299. It features a Titanium body."}]
    print("Extracting...")
    res = await llm_service.extract_products(chunks)
    print("Result:", res)

asyncio.run(main())
