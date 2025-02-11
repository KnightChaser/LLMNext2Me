# langchain_CacheBackedEmbeddings_with_tracing.py
"""
This example demonstrates how to cache document embeddings using LangChain’s CacheBackedEmbeddings
while also tracking the embedding procedure with the LangSmith client. The embedding and
indexing of documents is wrapped in a function and decorated with the LangSmith @traceable
decorator.
"""

import os
import time
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker

# Load API credentials and initialize the LangSmith client.
get_api_credentials()
langsmith_client: Client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Create a cache-backed embedding storage.
embedding = OpenAIEmbeddings()
store = LocalFileStore(
    "/home/knightchaser/cache/LLMNext2Me"  # Modify this path with your own path.
)
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,
)

# Load the documents and split them into chunks.
raw_documents = TextLoader("./resource/sample_sales_report.txt").load()

text_splitter = SemanticChunker(
    # Initialize SemanticChunker with OpenAIEmbeddings.
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5,
)
documents = text_splitter.split_documents(raw_documents)


# Wrap the embedding & indexing code in a function decorated for tracing.
@traceable(client=langsmith_client)
def embed_and_index_documents(docs, embedder):
    """
    Embeds the documents using the provided embedder and indexes them with FAISS.
    This function is decorated with LangSmith's @traceable so that its execution is tracked.
    """
    return FAISS.from_documents(docs, embedder)


# Execute the embedding procedure twice.
for iteration in range(2):
    start_time = time.time()
    print(f"Iteration {iteration + 1}:")
    print(f"Database entries (empty if initially created): {list(store.yield_keys())}")

    # Call the decorated function to embed and index the documents.
    db = embed_and_index_documents(documents, cached_embedder)

    elapsed_time = time.time() - start_time
    print(f"Embedding and storing the documents took {elapsed_time:.6f} seconds.")
    print(f"Number of documents: {len(documents)}\n")
