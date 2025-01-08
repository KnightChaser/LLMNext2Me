# ollama_basic.py
# Assume ollama3 and llama3:latest is already installed.
import os
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate 

get_api_credentials()

# Initialize LangSmith client
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

prompt = ChatPromptTemplate.from_template("Explain about the given topic briefly(<300 characters): {topic}")

llm = ChatOllama(model = "llama3:latest")

chain = prompt | llm | StrOutputParser()

@traceable
def ask_topic(topic: str) -> str:
    answer = chain.invoke({"topic": topic})
    return answer

if __name__ == "__main__":
    topic = input("Enter the topic: ")
    print(f"Answer: {ask_topic(topic)}")
