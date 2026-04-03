"""
query_engine.py — LlamaIndex Query Engine demo
Shows how a query engine wraps a retriever + LLM to produce a full answer.
Runs multiple queries so students can see different docs retrieved + different answers each time.
Uses qwen3-80b via our academy server.
"""

import requests
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse, CompletionResponseGen
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ── Custom LLM ────────────────────────────────────────────────────────────────

class UKISAILlm(CustomLLM):
    model: str = "qwen3-80b"
    api_url: str = "https://api.ukisai.academy/chat"
    system_prompt: str = "You are a helpful assistant. Answer concisely based on the provided context."

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


# ── Setup ─────────────────────────────────────────────────────────────────────

Settings.llm = UKISAILlm()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = [
    Document(text="Weather is nice outside."),
    Document(text="Dogs and cats are animals."),
    Document(text="Retrievers help us retrieve relevant information."),
    Document(text="Pilots operate airplanes."),
    Document(text="Madrid is the capital of Spain."),
    Document(text="LlamaIndex uses retrievers and query engines to answer questions."),
]

print("=== Building index — 6 documents loaded ===\n")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)


# ── Helper ────────────────────────────────────────────────────────────────────

def run_query(query: str):
    print("=" * 60)
    print(f"  Query: \"{query}\"")
    print("=" * 60)

    response = query_engine.query(query)

    print("  Retrieved sources:")
    for src in response.source_nodes:
        print(f"    [{src.score:.4f}]  {src.node.text}")

    print(f"\n  Answer (qwen3-80b): {response}\n")


# ── Four queries ──────────────────────────────────────────────────────────────

run_query("What is the weather like?")
run_query("What kind of animals are pets?")
run_query("Who flies planes?")
run_query("How does LlamaIndex find information?")
