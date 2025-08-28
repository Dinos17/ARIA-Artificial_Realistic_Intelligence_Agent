# aria_llm.py
import os
from dotenv import load_dotenv
import asyncio
import httpx
import itertools
from personality_aria import get_personality_prompt

# Load environment variables from .env
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "moonshotai/kimi-k2:free")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 45))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

# Load all OpenRouter API keys
KEYS = [os.getenv(f"OPENROUTER_API_KEY_{i}") for i in range(1, 11)]
KEYS = [k for k in KEYS if k]

# Ensure fallback key is included
fallback_key = os.getenv("OPENROUTER_API_KEY")
if fallback_key and fallback_key not in KEYS:
    KEYS.append(fallback_key)

if not KEYS:
    raise RuntimeError("No OpenRouter API keys found in environment variables.")

key_cycle = itertools.cycle(KEYS)

class ARIALLM:
    def __init__(self, model_id: str = None):
        self.model_id = model_id or MODEL_ID
        self.current_key = next(key_cycle)

    async def acomplete(self, user_input: str, chat_history: list, user_name: str, summary: str = "") -> str:
        system_prompt = get_personality_prompt(user_name)
        if summary:
            system_prompt += f"\n\nMemory summary: {summary}"

        payload = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system_prompt}]
                        + chat_history
                        + [{"role": "user", "content": user_input}]
        }

        retries = 0
        while retries <= MAX_RETRIES:
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    headers = {"Authorization": f"Bearer {self.current_key}"}
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=payload, headers=headers
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 429):
                    self.current_key = next(key_cycle)
                retries += 1
            except Exception:
                retries += 1

        return "[Error: Failed after retries]"
