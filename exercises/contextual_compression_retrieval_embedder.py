# contextual_compression_retrieval_embedder.py
"""
Embeddings Filter Retrieval Demo

This script demonstrates how to create a FAISS vector store from semantically split
document chunks, then retrieve documents using an embeddings-based filter to remove
unnecessary passages before LLM response generation. Unlike the LLM-based compression,
this approach uses only embedding computations to improve responsiveness and reduce cost.
LangSmith tracking is applied via the @traceable decorator.
"""

import os
import warnings

from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter


def pretty_print_documents(documents) -> None:
    """Prints each document's content, separated by a line."""
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document #{i+1}:\n\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )
    )


# -----------------------------------------------------------------------------
# Step 0: Initialize Credentials and LangSmith Client
# -----------------------------------------------------------------------------
get_api_credentials()
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Step 1: Setup CacheBackedEmbeddings with OpenAIEmbeddings
# -----------------------------------------------------------------------------
# We use a specific embedding model for document vectorization.
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
cache_dir = "/home/knightchaser/cache/LLMNext2Me"  # Adjust as needed.
os.makedirs(cache_dir, exist_ok=True)
store = LocalFileStore(cache_dir)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,
)

# -----------------------------------------------------------------------------
# Step 2: Load Documents using TextLoader
# -----------------------------------------------------------------------------
document_paths = [
    "./resource/estra_accident_report.txt",
    "./resource/estra_financial_report.txt",
]

documents = []
for path in document_paths:
    loader = TextLoader(path)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from '{path}'.")
    documents.extend(docs)

# -----------------------------------------------------------------------------
# Step 3: Split Documents using SemanticChunker
# -----------------------------------------------------------------------------
# Instead of a simple character-based splitter, we use a semantic chunker.
chunker = SemanticChunker(
    OpenAIEmbeddings(model="text-embedding-ada-002"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)
split_documents = chunker.split_documents(documents)
print(f"Split documents into {len(split_documents)} chunks.")

# -----------------------------------------------------------------------------
# Step 4: Create FAISS Vector Store from Document Chunks
# -----------------------------------------------------------------------------
print("Creating FAISS vector store...")
faiss_db = FAISS.from_documents(split_documents, cached_embedder)
print("FAISS vector store created successfully.")

# -----------------------------------------------------------------------------
# Step 5: Create a Base Retriever using FAISS (similarity search)
# -----------------------------------------------------------------------------
# We use the FAISS retriever with a simple similarity search.
faiss_retriever = faiss_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)

# -----------------------------------------------------------------------------
# Step 6: Setup the Embeddings Filter Compressor
# -----------------------------------------------------------------------------
# The EmbeddingsFilter uses our cached embedder to filter documents based on a similarity threshold.
embeddings_filter = EmbeddingsFilter(
    embeddings=cached_embedder,  # Uses the same embedder for filtering.
    similarity_threshold=0.6,  # Adjust the threshold as needed.
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=faiss_retriever,
)


# -----------------------------------------------------------------------------
# Step 7: Wrap Retrieval Functions with LangSmith Tracking
# -----------------------------------------------------------------------------
@traceable
def get_base_documents(query: str):
    """Retrieve documents using the base FAISS retriever."""
    # Using .invoke() per new API recommendations.
    return faiss_retriever.invoke(query)


@traceable
def get_compressed_documents(query: str):
    """Retrieve documents using the embeddings-based compression retriever."""
    return compression_retriever.invoke(query)


# -----------------------------------------------------------------------------
# Step 8: Interactive Query Loop
# -----------------------------------------------------------------------------
print("\nEnter your queries below. Type 'exit' to quit.")
while True:
    user_prompt = input("\nEnter your query (or 'exit' to quit): ").strip()
    if user_prompt.lower() == "exit":
        break

    base_documents = get_base_documents(user_prompt)
    print(f"# of documents returned by base retriever: {len(base_documents)}")

    compressed_documents = get_compressed_documents(user_prompt)
    print(
        f"# of documents returned by compression retriever: {len(compressed_documents)}"
    )

    pretty_print_documents(compressed_documents)
