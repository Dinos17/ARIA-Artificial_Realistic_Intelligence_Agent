# main.py
import asyncio
from aria_llm import ARIALLM
from aria_memory import ARIAMemory
from dotenv import load_dotenv
import re

# Load .env file
load_dotenv()

async def main():
    memory = ARIAMemory()

    # Ask for username only if not already stored
    if not memory.user_name:
        user_name = input("Hello! What is your name? ").strip()
        if user_name:
            memory.user_name = user_name
            memory.personal_regex = re.compile(re.escape(user_name), re.IGNORECASE)
            memory.store_username(user_name)

    # Initialize LLM
    llm = ARIALLM()

    print(f"\nARIA online. Type 'exit' to shut me down, {memory.user_name}.\n")

    while True:
        try:
            user_input = input(f"{memory.user_name}: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Shutting down ARIA...")
                break

            # Retrieve context (short-term + long-term)
            context = memory.get_context(top_k=5, filter_personal=True)

            # Query LLM
            response = await llm.acomplete(
                user_input,
                chat_history=context["chat_history"],
                user_name="[REDACTED]",  # Never send actual username
                summary=context["summary"]
            )
            print(f"ARIA: {response}\n")

            # Update memories
            memory.update(user_input, response)
            memory.update_summary(" ".join([msg["content"] for msg in memory.chat_history[-20:]]))

        except KeyboardInterrupt:
            print("\nShutting down ARIA...")
            break
        except Exception as e:
            print(f"[Error] {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
