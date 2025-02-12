# faiss_text_db.py
"""
FAISS is a database that allows for fast similarity searches on large collections of vectors. This script demonstrates how to use FAISS to create a text database and perform similarity searches on it. In this version, we convert the FAISS index to a retriever that uses the MMR (Maximum Marginal Relevance) search technique.
"""

import os
import warnings
from credentials.get_api_credentials import get_api_credentials
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores.faiss import FAISS

# Load environment variables (e.g., for API keys)
get_api_credentials()

# Set up cache directories and suppress warnings.
os.environ["HF_HOME"] = "/home/knightchaser/cache/HuggingFace"  # Modify as needed
warnings.filterwarnings("ignore")

# Step 1: Setup CacheBackedEmbeddings with OpenAIEmbeddings
embedding = OpenAIEmbeddings()
cache_dir = "/home/knightchaser/cache/LLMNext2Me"  # Modify as needed.
os.makedirs(cache_dir, exist_ok=True)
store = LocalFileStore(cache_dir)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,
)

# Step 2: Load and Semantically Split the Documents
# Below two documents are imaginary reports for demonstration purposes.
document_paths = [
    "./resource/estra_accident_report.txt",
    "./resource/estra_financial_report.txt",
]

# Create a SemanticChunker instance.
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)

all_chunks = []
for path in document_paths:
    loader = TextLoader(path)
    raw_docs = loader.load()
    print(f"Loaded '{path}' with {len(raw_docs)} document(s).")
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Document '{path}' split into {len(chunks)} chunks.")
    all_chunks.extend(chunks)

# Step 3: Create a FAISS Database from the Document Chunks
print("Creating FAISS database from the document chunks...")
faiss_db = FAISS.from_documents(all_chunks, cached_embedder)
print("FAISS database created successfully.")

# Step 4: Convert the FAISS database to a retriever using MMR search technique.
# Here, we use MMR to rerank the results for diversity.
retriever = faiss_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,  # Number of final results to return.
        "fetch_k": 10,  # Number of candidates to consider for MMR.
        "lambda_mult": 0.5,  # Trade-off parameter between relevance and diversity.
    },
)
print("FAISS retriever configured with MMR search.")

# Step 5: Perform an Interactive MMR Similarity Search
while True:
    query = input("Enter a query for MMR similarity search (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    results = retriever.get_relevant_documents(query)
    print("\nMMR Similarity Search Results:")
    for idx, res in enumerate(results):
        snippet = res.page_content
        print(f"\nResult {idx+1}:")
        print(f"Snippet: {snippet}...")
        print(f"Metadata: {res.metadata}")
    print("\n")
