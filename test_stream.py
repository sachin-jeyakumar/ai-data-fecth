import asyncio
from ollama import AsyncClient

async def main():
    print("Testing pure ollama stream...")
    client = AsyncClient(host="http://localhost:11434")
    try:
        stream = await client.chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "Hello, write a very short sentence."}],
            stream=True
        )
        async for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print("\nSuccess!")
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
