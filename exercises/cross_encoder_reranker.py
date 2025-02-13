# cross_encoder_reranker.py
"""
Cross encoder reranker is a re-ranking model that takes a list of candidate pairs and re-ranks them based on their similarity. By doing so, it can improve the performance of information retrieval systems by ensuring that the most relevant documents are ranked higher in the search results.
"""

import os
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


get_api_credentials()
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])


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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-distilbert-dot-v5"
)
retriever = FAISS.from_documents(texts, embeddings).as_retriever(
    search_kwargs={"k": 10}
)

# Initialize the reranker model
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# Prepare the compressor
compressor = CrossEncoderReranker(model=model, top_n=3)
compressor_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


@traceable
def ask_llm(query: str) -> None:
    compressed_documents = compressor_retriever.invoke(query)

    pretty_print_documents(compressed_documents)


ask_llm("What does SOC(Security Operations Center) usually do?")
