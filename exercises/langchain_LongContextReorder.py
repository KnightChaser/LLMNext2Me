# langchain_LongContextReorder.py
"""
LLMs tend to have a long context window, which can be useful for document retrieval tasks.
However, as the input size increases, they tend to focus on first and last tokens, which can
skip away the information at the middle of the document. This script demonstrates how to
reorder the context window to focus on the middle of the document, which can be useful for
enhancing the retrieval performance.
"""

import os
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

get_api_credentials()
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "How AI technology is changing the world?"
texts = [
    # Suppose this is a bunch of documents retrieved from a database.
    # With a high probability, there might be a lot of noise in the middle of the document.
    "AI technology is changing the world in many ways. It is being used in various sectors such as finances, educations, medical, etc.",
    "During the COVID-19 pandemic, AI helped to estimate the spread of diseases",
    "A random small article",
    "Bitcoin is one of the most popular cryptocurrencies, enabling a new way of payment",
    "ChatGPT is helping people worldwide to communicate with each other in a more efficient way, proving the success of AI in the communication sector",
    "Linux is one of the most popular operating systems, used by many developers worldwide",
    "AI is now integrated with mobile devices, to reduce energy consumption, suggest new features to users, and empower mobile security features.",
    "Sorting algorithms are used to sort data in a more efficient way, reducing the time complexity of the algorithm",
    "It is expected that there will be more than 50 billion IoT devices by 2030, which will generate a huge amount of data",
    "AI is now being used in the agriculture sector to predict the weather, suggest the best time to plant crops, and reduce the use of pesticides",
    "There are noise and irrelevant information in the middle of the document",
]

retriever = Chroma.from_texts(texts=texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)


# Just print the retrieved documents for now.
def pretty_print_documents(documents) -> None:
    """Prints a separator-delimited string for each document's content."""
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document #{i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]
        )
    )


print("=== Before Reordering ===")
documents = retriever.invoke(query)
pretty_print_documents(documents)
print("\n" * 10)

print("=== After Reordering ===")
reordered_retriever = LongContextReorder()
reordered_documents = reordered_retriever.transform_documents(documents)
pretty_print_documents(reordered_documents)


# Based on the context reordering, we can create the question-answer pair,
# expecting enhanced response generation performance.
def reorder_documents(documents):
    def format_documents(documents):
        return "\n".join(
            [f"Document #{i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]
        )

    reordered_retriever = LongContextReorder()
    reordered_documents = reordered_retriever.transform_documents(documents)
    return format_documents(reordered_documents)


@traceable
def ask_llm(question: str) -> str:
    template = """Given this text extracts:
    {context}
    
    -----
    Please answer the following question:
    {question}
    
    Answer in the following languages: {language}
    """

    # Define prompt and chain
    prompt = ChatPromptTemplate.from_template(template)

    llm_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | RunnableLambda(reorder_documents),
            "question": itemgetter("question"),
            "language": itemgetter("language"),
        }
        | prompt
        | ChatOpenAI(model="gpt-3.5-turbo")
        | StrOutputParser()
    )

    answer = llm_chain.invoke({"question": question, "language": "en"})
    return answer


print("\n\n")
print(f"LLM's answer: {ask_llm(query)}")
