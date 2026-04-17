import asyncio
import os
from pathlib import Path
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from tavily import AsyncTavilyClient

SKILLS_FILE = Path("SKILLS.md")
LLM_MODEL = "qwen3-80b"

# ── 1. Sistemski Prompt (Ličnost i Granice) ──
SYSTEM_PROMPT = """Ti si ekskluzivni AI asistent za kompaniju UkisAI. 
Tvoja JEDINA svrha i jedini domen znanja je UkisAI platforma, njene usluge, tim i funkcionalnosti.

Pravila ponašanja:
1. Odgovaraj samo na pitanja koja su direktno vezana za UkisAI.
2. Ako korisnik postavi pitanje koje NIJE u vezi sa UkisAI (npr. vremenska prognoza, pisanje koda, recepti), ODBIJ odgovor i reci da si tu isključivo za UkisAI podršku.
3. Budi profesionalan i koncizan."""

# ── 2. Alati ──
async def search_web(query: str) -> str:
    """Pretraga interneta za najnovije informacije o UkisAI."""
    client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return str(await client.search(query))

def load_skills() -> str:
    """Učitava lokalno znanje iz SKILLS.md fajla."""
    if SKILLS_FILE.exists():
        return SKILLS_FILE.read_text(encoding="utf-8")
    return "(Nema lokalnih informacija)"

# ── 3. Inicijalizacija LLM-a ──
llm = OpenAILike(
    model=LLM_MODEL,
    api_base="https://api.ukisai.academy/v1",
    api_key="dummy",
    is_function_calling_model=True,
    is_chat_model=True,
    context_window=32000,
)

final_system_prompt = SYSTEM_PROMPT + "\n\nOvo su interne informacije koje znaš:\n" + load_skills()

# ── 4. Sklapanje Agenta ──
agent = FunctionAgent(
    tools=[search_web],
    llm=llm,
    system_prompt=final_system_prompt,
)

# ── 5. Chat interfejs ──
async def chat():
    print("============================================================")
    print(" UkisAI Chatbot inicijalizovan. (Unesi 'exit' za izlaz)")
    print("============================================================")
    
    ctx = Context(agent) # Pamti istoriju konverzacije
    
    while True:
        user_input = input("\nTi: ")
        if user_input.lower() in ['exit', 'quit', 'izlaz']:
            print("Izlaz iz chat-a...")
            break
            
        response = await agent.run(user_msg=user_input, ctx=ctx)
        print(f"\nUkisAI Agent: {response}")

if __name__ == "__main__":
    asyncio.run(chat())