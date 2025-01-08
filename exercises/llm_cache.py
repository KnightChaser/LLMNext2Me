# llm_cache.py
# By utilizing cache in LLM application, we can reduce the number of API calls to the LangSmith API and OpenAI API.
import os
import time
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

get_api_credentials()

langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Create the LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

prompt = PromptTemplate.from_template("Briefly introduce about the country named {country_name}")
chain = prompt | llm

# Set up the In-Memory cache for the efficient LLM workflow
set_llm_cache(InMemoryCache())

@traceable
def ask_about_country(country_name: str):
    return chain.invoke({"country_name": country_name})

if __name__ == "__main__":
    country_name = input("Enter the country name: ")

    # First time, it will make an API call to the LangSmith API and OpenAI API
    start_time = time.time()
    print(f"1st hit: {ask_about_country(country_name).content}")
    end_time = time.time()
    print(f" => Time taken: {end_time - start_time:.6f} seconds")

    # Second time, it will use the cache to retrieve the response
    start_time = time.time()
    print(f"2nd hit: {ask_about_country(country_name).content}")
    end_time = time.time()
    print(f" => Time taken: {end_time - start_time:.6f} seconds")
