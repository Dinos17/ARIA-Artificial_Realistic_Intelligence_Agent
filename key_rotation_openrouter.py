# key_rotation_openrouter.py
import os
import itertools
import asyncio
import httpx

MODEL_ID = os.getenv("MODEL_ID", "moonshotai/kimi-k2:free")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 45))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

# Collect keys from .env
keys = [os.getenv(f"OPENROUTER_API_KEY_{i}") for i in range(1, 11)]
keys = [k for k in keys if k]

async def query_model(prompt: str, api_key: str):
    payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                resp = await client.post("https://openrouter.ai/api/v1/chat/completions",
                                         json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                return None  # Rate limit hit, try next key
            retries += 1
        except Exception:
            retries += 1
    return "[Error: Failed after retries]"

async def main():
    prompts = ["Hello ARIA", "Tell me a joke", "What is the weather today?"]
    key_cycle = itertools.cycle(keys)

    for prompt in prompts:
        result = None
        attempts = 0
        while result is None and attempts < len(keys):
            key = next(key_cycle)
            result = await query_model(prompt, key)
            attempts += 1
        print(f"Prompt: {prompt}\nResponse: {result}\n")

if __name__ == "__main__":
    asyncio.run(main())
