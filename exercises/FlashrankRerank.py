# FlashrankReranker.py
"""
FlashrankReranker is a re-ranking model based on the Flashrank algorithm. It takes a list of candidate pairs and re-ranks them based on their similarity using the Flashrank algorithm. By doing so, it can improve the performance of information retrieval systems by ensuring that the most relevant documents are ranked higher in the search results.

It's super fast and efficient, making it ideal for real-time applications where speed is crucial. It even doesn't need GPU, Torch, or any other heavy dependencies. Just pure Python code.
"""

import os
import warnings
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI

# Base setup
get_api_credentials()
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
warnings.filterwarnings("ignore")


def pretty_print_documents(documents) -> None:
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document #{i+1}:\n\n" + d.page_content for i, d in enumerate(documents)]
        )
    )


# Load, process and store documents
documents = TextLoader("./resource/beginner_IT_terminology_list.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=25)
texts = text_splitter.split_documents(documents)

# Add unique ID for each text
for index, text in enumerate(texts):
    print(f"Text ID: {index} is now being processed.")
    text.metadata["id"] = index

# Create the retriever based on FAISS vectorstores
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever(
    search_kwargs={"k": 10}
)

# Create LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Prepare the document compressor retriever
compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


@traceable
def ask_llm(query: str) -> None:
    compressed_documents = compression_retriever.invoke(query)

    print(f"# of selected documents: {len(compressed_documents)}")
    print(
        f"Selected document No.: {[documents.metadata['id'] for documents in compressed_documents]}"
    )
    pretty_print_documents(compressed_documents)


ask_llm("What does SOC(Security Operations Center) usually do?")

# # of selected documents: 3
# Selected document No.: [126, 125, 123]
# Document #1:
#
# A security operations center (SOC) can refer to the team of experts that monitor an organization’s ability to operate securely. It can also refer to
# ----------------------------------------------------------------------------------------------------
# Document #2:
#
# Security Operations Center (SOC)
# ----------------------------------------------------------------------------------------------------
# Document #3:
#
# A security architect is someone who develops and maintains the security of an organization’s network. They also collaborate with business leaders,
