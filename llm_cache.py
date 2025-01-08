# llm_cache.py
# By utilizing cache in LLM application, we can reduce the number of API calls to the LangSmith API and OpenAI API.
import os
import time
from dotenv import load_dotenv
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

load_dotenv()

openai_api_key      = os.getenv("OPENAI_API_KEY", "")
langsmith_api_key   = os.getenv("LANGSMITH_API_KEY", "")

if not openai_api_key and not openai_api_key.startswith("sk-"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")
if not langsmith_api_key and not langsmith_api_key.startswith("lsv2-"):
    raise ValueError("LANGSMITH_API_KEY is not set in the environment variables")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

# Initialize LangSmith client
langsmith_client = Client(api_key=langsmith_api_key)

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
