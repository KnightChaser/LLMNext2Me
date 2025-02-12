# contextual_compression_retrieval.py
"""
Contextual Compression Retrieval with Auditing and Semantic Splitting

This script demonstrates how to create a FAISS vector store from semantically split
document chunks, then retrieve documents with contextual compression. The LLM-based
compression steps are audited via LangSmith using the @traceable decorator, and we
avoid deprecated retrieval methods by using `.invoke()` instead of `.get_relevant_documents()`.
"""

import os
import warnings
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor

# -----------------------------------------------------------------------------
# Step 0: Initialize Credentials and LangSmith Client
# -----------------------------------------------------------------------------
get_api_credentials()
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
warnings.filterwarnings("ignore")


def pretty_print_documents(documents) -> None:
    """Prints a separator-delimited string for each document's content."""
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document #{i+1}:\n\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )
    )


# -----------------------------------------------------------------------------
# Step 1: Setup CacheBackedEmbeddings with OpenAIEmbeddings
# -----------------------------------------------------------------------------
# Use a specific model for embedding.
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
cache_dir = "/home/knightchaser/cache/LLMNext2Me"  # Modify as needed.
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
# Instead of a simple character splitter, we use a semantic splitter.
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
# Note: We use the new .invoke() method (instead of .get_relevant_documents()) to avoid deprecation warnings.
faiss_retriever = faiss_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)

# -----------------------------------------------------------------------------
# Step 6: Setup LLM for Compression and Create the Compressor
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
compressor = LLMChainExtractor.from_llm(llm=llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=faiss_retriever,
)


# -----------------------------------------------------------------------------
# Step 7: Wrap Retrieval Functions with LangSmith Tracking
# -----------------------------------------------------------------------------
@traceable
def get_base_documents(query: str):
    """Retrieve documents using the base FAISS retriever."""
    return faiss_retriever.invoke(query)


@traceable
def get_compressed_documents(query: str):
    """Retrieve documents using the contextual compression retriever."""
    return compression_retriever.invoke(query)


# -----------------------------------------------------------------------------
# Step 8: Interactive Query Loop
# -----------------------------------------------------------------------------
print("\nEnter your queries below. Type 'exit' to quit.")
while True:
    user_prompt = input("\nEnter a query (or 'exit' to quit): ").strip()
    if user_prompt.lower() == "exit":
        break

    # Retrieve documents using the base retriever.
    base_documents = get_base_documents(user_prompt)
    print(f"# of documents returned by base retriever: {len(base_documents)}")

    # Retrieve compressed documents via the compression retriever.
    compressed_documents = get_compressed_documents(user_prompt)
    print(
        f"# of documents returned by compression retriever: {len(compressed_documents)}"
    )

    # Print out the compressed documents.
    pretty_print_documents(compressed_documents)

# Loaded 1 document(s) from './resource/estra_accident_report.txt'.
# Loaded 1 document(s) from './resource/estra_financial_report.txt'.
# Split documents into 14 chunks.
# Creating FAISS vector store...
# FAISS vector store created successfully.
#
# Enter your queries below. Type 'exit' to quit.
#
# Enter a query (or 'exit' to quit): Which accident invovled the company Estra happened in the year 2063?
# # of documents returned by base retriever: 5
# # of documents returned by compression retriever: 2
# Document #1:
#
# 2063-EX-STRATO-001
# ----------------------------------------------------------------------------------------------------
# Document #2:
#
# The stratospheric explosion events on April 5, 2063
#
# Enter a query (or 'exit' to quit): How the accident invovled the company Estra which happened at April 5, 2063 happened?
# # of documents returned by base retriever: 5
# # of documents returned by compression retriever: 2
# Document #1:
#
# On April 5, 2063, a series of explosions were recorded in the stratospheric operational zone above the Horizon Spaceport, resulting in extensive infrastructural damage and significant operational loss.
# ----------------------------------------------------------------------------------------------------
# Document #2:
#
# The stratospheric explosion events on April 5, 2063, underscore the critical importance of rigorous cybersecurity protocols in complex aerospace systems.
