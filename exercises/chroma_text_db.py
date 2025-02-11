# chroma_text_db_with_semantic_chunking.py
"""
This script demonstrates a basic workflow using Chroma with LangChain,
utilizing CacheBackedEmbeddings and SemanticChunker to semantically
split text documents into chunks before embedding them.

Workflow:
1. Load two example news articles from the "./resource" directory.
2. Semantically split each article into multiple chunks.
3. Embed the chunks using CacheBackedEmbeddings (with OpenAIEmbeddings as the underlying model).
4. Create a Chroma database from the chunks of the first article.
5. Add the chunks of subsequent articles to the same Chroma database.
6. Perform an interactive similarity search on the Chroma database.
"""

import os
import warnings
from typing import Union
from pprint import pprint
from dotenv import load_dotenv
from credentials.get_api_credentials import get_api_credentials
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

# Load environment variables (e.g., for API keys)
load_dotenv()
get_api_credentials()

# Set up cache directories and suppress warnings.
os.environ["HF_HOME"] = "/home/knightchaser/cache/HuggingFace"  # Modify as needed
warnings.filterwarnings("ignore")

# Step 1: Setup CacheBackedEmbeddings with OpenAIEmbeddings ===
embedding = OpenAIEmbeddings()
cache_dir = "/home/knightchaser/cache/LLMNext2Me"  # Modify this path as needed.
os.makedirs(cache_dir, exist_ok=True)
store = LocalFileStore(cache_dir)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,
)

# Step 2: Load and Semantically Split the Documents ===
# Define a list of paths to the example news articles.
document_paths = [
    "./resource/sample_news_article_1.txt",
    "./resource/sample_news_article_2.txt",
]

# Create a SemanticChunker instance.
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)

# Iterate over the document paths to load and split them.
db: Union[Chroma, None] = None
for idx, path in enumerate(document_paths):
    loader = TextLoader(path)
    raw_docs = loader.load()
    print(f"Loaded '{path}' with {len(raw_docs)} document(s).")

    # Split the raw documents into semantically coherent chunks.
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Document '{path}' split into {len(chunks)} chunks.")

    if idx == 0:
        # Step 3: Create a Chroma Database from the first document's chunks.
        print("Creating Chroma database from the first document chunks...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=cached_embedder,
            collection_name="news_articles",
        )
        print("Chroma database created successfully.")
    else:
        # Step 4: Add subsequent document chunks to the existing Chroma database.
        assert db is not None
        print(
            f"Adding document chunks from '{path}' to the existing Chroma database..."
        )
        db.add_documents(chunks)
        print(f"Document chunks from '{path}' added successfully.")

assert db is not None
print("\nAll documents currently in the Chroma collection:")
all_docs = db.get()
pprint(all_docs)
print("\n")

# Step 5: Perform an Interactive Similarity Search ===
while True:
    query = input("Enter a query (or 'exit' to quit): ")
    if query.lower() == "exit":
        print("Exiting. The database will be reset!")
        db.reset_collection()
        break

    print(f"\nSearching for: '{query}'")
    results = db.similarity_search(query=query, k=3)
    for i, res in enumerate(results):
        # Show a snippet of the matched chunk.
        snippet = res.page_content
        print(f"\nResult #{i+1}:")
        print(f"Snippet: {snippet}...")
        print(f"Metadata: {res.metadata}")
    print("\n")
