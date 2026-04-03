"""
retriever.py — LlamaIndex Retriever demo
Shows how the retriever pulls ONLY the relevant documents for each query,
even when the index contains completely unrelated topics.
"""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── 0. Local embedding model — no API key needed ──────────────────────────────
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ── 1. A mix of completely unrelated documents ────────────────────────────────
documents = [
    Document(text="Weather is nice outside."),
    Document(text="Dogs and cats are animals."),
    Document(text="Retrievers help us retrieve relevant information."),
    Document(text="Pilots operate airplanes."),
    Document(text="Madrid is the capital of Spain."),
    Document(text="LlamaIndex uses retrievers and query engines to answer questions."),
]

print("=== Building index — 6 unrelated documents loaded ===\n")
index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)


# ── Helper to print results cleanly ──────────────────────────────────────────
def run_query(query: str):
    print(f"Query: \"{query}\"")
    print("-" * 55)
    nodes = retriever.retrieve(query)
    for node in nodes:
        print(f"  [{node.score:.4f}]  {node.node.text}")
    print()


# ── 2. Four very different queries — watch which docs get pulled each time ────

run_query("What is the weather like?")

run_query("What kind of animals are pets?")

run_query("Who flies planes?")

run_query("How does LlamaIndex find information?")


print("=" * 55)
print("Notice: each query pulled DIFFERENT documents from the same index.")
print("The retriever matched by meaning, not keywords.")
print("No LLM was called — just vector similarity search.")
