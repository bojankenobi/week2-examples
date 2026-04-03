"""
workflow.py — LlamaIndex Workflow demo
Two-step workflow: generate a joke, then critique it.
Uses qwen3-80b via our academy server instead of OpenAI.
"""

import asyncio
import requests
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.core.llms.callbacks import llm_completion_callback


# ── Custom LLM — calls our academy server ─────────────────────────────────────

class UKISAILlm(CustomLLM):
    model: str = "qwen3-80b"
    api_url: str = "https://api.ukisai.academy/chat"
    system_prompt: str = "You are a helpful assistant."

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32000,
            num_output=1024,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "message": prompt,
        }
        resp = requests.post(self.api_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("response")
            or data.get("message")
            or data.get("content")
            or data.get("text")
            or str(data)
        )
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        result = self.complete(prompt, **kwargs)
        def _gen():
            yield result
        return _gen()

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return await asyncio.to_thread(self.complete, prompt, **kwargs)


# ── Events ────────────────────────────────────────────────────────────────────

class JokeEvent(Event):
    joke: str

class JokeResult(StopEvent):
    joke: str
    critique: str


# ── Workflow ──────────────────────────────────────────────────────────────────

class JokeFlow(Workflow):
    llm = UKISAILlm()

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> JokeResult:
        joke = ev.joke
        prompt = f"Give a thorough analysis and critique of the following joke: {joke} keep it under 500 words"
        response = await self.llm.acomplete(prompt)
        return JokeResult(joke=joke, critique=str(response))


async def main():
    topic = input("What do you want to joke about? ").strip()
    if not topic:
        topic = "cats"

    w = JokeFlow(timeout=60, verbose=False)
    print("=" * 60)
    print(f"  TOPIC: {topic}")
    print("=" * 60)

    result = await w.run(topic=topic)

    print(f"\n  Joke     : {result.joke}")
    print(f"\n  Critique : {result.critique}")

asyncio.run(main())
