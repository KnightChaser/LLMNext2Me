#!/usr/bin/env python3
"""
FAISS VectorStore Retrieval Demo with Separate Document and Query Embeddings

This script demonstrates:
  1. Loading documents and splitting them into semantic chunks.
  2. Creating a FAISS vector store using cached document embeddings.
  3. Overriding the query embedding function with a different model.
  4. Retrieving documents via both standard similarity search and MMR (Maximum Marginal Relevance).

Ensure that your API credentials are set up via get_api_credentials().
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


def main():
    # Step 0: Load API credentials and set up environment.
    get_api_credentials()
    os.environ["HF_HOME"] = "/home/knightchaser/cache/HuggingFace"  # Modify as needed.
    warnings.filterwarnings("ignore")

    # -------------------------------------------------------------------------
    # Step 1: Define Embedding Models for Documents and Queries
    # -------------------------------------------------------------------------
    # Document embedding model (for indexing and document splitting)
    doc_embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    cache_dir = "/home/knightchaser/cache/LLMNext2Me"  # Modify as needed.
    os.makedirs(cache_dir, exist_ok=True)
    store = LocalFileStore(cache_dir)
    cached_doc_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=doc_embedding_model,
        document_embedding_cache=store,
        namespace=doc_embedding_model.model,
    )

    # Query embedding model (for user queries).
    # IMPORTANT: The document and query embeddings need to be in the same space.
    query_embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # -------------------------------------------------------------------------
    # Step 2: Load and Semantically Split the Documents
    # -------------------------------------------------------------------------
    # Updated document paths to match your current situation.
    document_paths = [
        "./resource/estra_accident_report.txt",
        "./resource/estra_financial_report.txt",
    ]

    print("Loading and splitting documents...")
    # Use the document embedding model for semantic chunking.
    text_splitter = SemanticChunker(
        doc_embedding_model,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1.25,
    )

    all_chunks = []
    for path in document_paths:
        loader = TextLoader(path)
        raw_docs = loader.load()
        print(f"Loaded '{path}' with {len(raw_docs)} document(s).")
        chunks = text_splitter.split_documents(raw_docs)
        print(f"Document '{path}' split into {len(chunks)} chunk(s).")
        all_chunks.extend(chunks)

    # -------------------------------------------------------------------------
    # Step 3: Create the FAISS Vector Store
    # -------------------------------------------------------------------------
    print("Creating FAISS vector store from document chunks...")
    # Note: The document embeddings (from cached_doc_embedder) are computed using doc_embedding_model.
    faiss_db = FAISS.from_documents(all_chunks, cached_doc_embedder)
    print("FAISS vector store created successfully.")

    # Override the vector store's query embedding function to use the query model.
    # Make sure that the query embeddings are compatible with the document embeddings!
    faiss_db.embedding_function = query_embedding_model.embed_query

    # -------------------------------------------------------------------------
    # Step 4: Configure Retrievers for Standard Similarity and MMR Searches
    # -------------------------------------------------------------------------
    similarity_retriever = faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # Return top 3 similar documents.
    )

    mmr_retriever = faiss_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,  # Final number of results.
            "fetch_k": 10,  # Number of candidates considered before re-ranking.
            "lambda_mult": 0.5,  # Balance between relevance and diversity.
        },
    )

    # -------------------------------------------------------------------------
    # Step 5: Interactive Query Loop for Retrieval
    # -------------------------------------------------------------------------
    print("\nEnter your queries below. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            break

        # Choose retrieval mode.
        mode_choice = (
            input(
                "Choose retrieval mode - (1) Standard Similarity, (2) MMR [default 1]: "
            ).strip()
            or "1"
        )
        if mode_choice == "2":
            retriever = mmr_retriever
            mode_name = "MMR Search"
        else:
            retriever = similarity_retriever
            mode_name = "Standard Similarity Search"

        print(f"\nPerforming {mode_name} for query: '{query}'")
        results = retriever.get_relevant_documents(query)

        if not results:
            print("No relevant documents found.")
        else:
            print(f"Found {len(results)} document(s):")
            for idx, doc in enumerate(results):
                snippet = doc.page_content.strip().replace("\n", " ")
                snippet_display = (
                    snippet if len(snippet) <= 200 else snippet[:200] + "..."
                )
                print(f"\nResult {idx + 1}:")
                print(f"Snippet: {snippet_display}")
                print(f"Metadata: {doc.metadata}")

    print("Exiting retrieval demo.")


if __name__ == "__main__":
    main()
