# langchain_CacheBackedEmbeddings.py
"""
To avoid unnecessary duplicated embedding API calls, we can use the CacheBackedEmbeddings class.
It caches the embeddings of the text and documents in the local storage.
And when the same text or document is requested, it returns the embeddings from the cache.
Thus, it helps to reduce the number of API calls and save the cost.
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
    "/home/knightchaser/cache/LLMNext2Me"
)  # Modify this path with your own path.
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,
)

# Load the documents and split it into chunks, embed them, and store the embeddings.
raw_documents = TextLoader("./resource/sample_sales_report.txt").load()

text_splitter = SemanticChunker(
    # Use OpenAIEmbeddings to initialize the SemanticChunker
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5,
)

documents = text_splitter.split_documents(raw_documents)

# First, we embed the documents and store the embeddings.
# Second, we load the embeddings from the cache. We don't need to embed the documents again, but just immediately load the embeddings from the cache.
for _ in range(2):
    start_time = time.time()
    print(
        f"Database entries(empty if it's initially created): {list(store.yield_keys())}"
    )
    db = FAISS.from_documents(documents, cached_embedder)
    print(
        f"Embedding and storing the documents took {time.time() - start_time:.6f} seconds."
    )
    print(f"Number of documents: {len(documents)}")

# Database entries(empty if it's initially created): []
# Embedding and storing the documents took 0.476457 seconds.
# Number of documents: 5
# Database entries(empty if it's initially created): ['text-embedding-ada-0028f842ef1-7f6e-562d-a013-265661bec540', 'text-embedding-ada-00208eeab16-cda3-58a7-9a6f-82cd27bbfe93', 'text-embedding-ada-002f60b31aa-6d7c-512a-8781-6a3960078860', 'text-embedding-ada-00284310a7c-9334-5a82-9922-d7edf9c6893d', 'text-embedding-ada-0020b27d4f1-5139-531e-a857-1ade8b0fd2fe']
# Embedding and storing the documents took 0.001451 seconds.
# Number of documents: 5
