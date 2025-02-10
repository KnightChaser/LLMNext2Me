# pdf_local_rag.py
# A local and small RAG(Retrieval Augmented Generation) system with user-provided PDF documents.

import os
import argparse

from langchain_core.vectorstores import VectorStoreRetriever
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Make sure the credentials environment variables are set
get_api_credentials()

# Initialize LangSmith client
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])


def create_retriever(file_path: str) -> VectorStoreRetriever:
    """
    Load a PDF file, split it into chunks, create embeddings,
    and return a VectorStoreRetriever for searching.
    """
    # Load documents
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    return retriever


def embed_file(file_path: str) -> VectorStoreRetriever:
    """
    Embed the given PDF and return the retriever.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return create_retriever(file_path)


def format_document(document_list):
    """
    Format retrieved documents into a single string.
    """
    return "\n\n".join([doc.page_content for doc in document_list])


@traceable
def create_rag_llm_chain(retriever: VectorStoreRetriever):
    """
    Create a RAG (Retrieval Augmented Generation) chain with a retriever.
    """
    # Load a prompt file that uses placeholders {context} and {question}
    prompt = load_prompt("./prompts/pdf_local_rag.yaml", encoding="utf-8")

    # Configure the local Ollama model
    llm = ChatOllama(
        model="llama3:latest", temperature=0.0  # adjust to your local model name
    )

    # Construct a chain of Runnables:
    #   1. "context": feed the input question to 'retriever' -> 'format_document'
    #   2. "question": a simple passthrough of the user input
    #   3. The result is then fed into 'prompt' -> 'llm' -> 'StrOutputParser'
    chain = (
        {"context": retriever | format_document, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    parser = argparse.ArgumentParser(description="Local PDF-based RAG with LLM.")
    parser.add_argument("--pdf_path", type=str, help="Path to your PDF file.")
    args = parser.parse_args()

    # If no CLI argument given, prompt the user for a path
    pdf_path = args.pdf_path
    if not pdf_path:
        pdf_path = input("Enter the path to your PDF file: ").strip()

    # Create the retriever
    retriever = embed_file(pdf_path)

    # Create the RAG chain
    chain = create_rag_llm_chain(retriever)

    print("\nType your questions about the PDF. Type '/bye' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "/bye":
            print("Exiting... Goodbye!")
            break

        # Run the chain on the user input
        response = chain.invoke(user_input)

        print(f"LLM: {response}\n")


if __name__ == "__main__":
    main()

# (Used Microsoft Digital Defense Report 2024 Overview.pdf)
# Type your questions about the PDF. Type '/bye' to exit.
#
# You: What are main themes of the given PDF document?
# LLM: Based on the provided information, the main themes of the Microsoft Digital Defense Report 2024 Overview appear to be:
#
# 1. The evolving cyber threat landscape and its implications for security.
# 2. The role of artificial intelligence (AI) in cybersecurity and its potential impact.
# 3. The importance of centering organizations on security and preserving privacy.
# 4. Strategic approaches to cybersecurity, including collective action and supporting the ecosystem.
#
# These themes are evident throughout the report's introduction, key developments, and actionable insights sections.
#
# You:
