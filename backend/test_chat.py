import asyncio
from services import llm_service

async def main():
    try:
        print("Starting chat...")
        chunks = [{"filename": "test.pdf", "page": 1, "text": "This is a test document."}]
        async for token in llm_service.chat_with_context("What is this?", chunks):
            print(token, end="", flush=True)
        print("\nDone!")
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
