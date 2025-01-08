
# get_api_credentials.py
# Boilerplate code to load the environment variables from the .env file

import os
from dotenv import load_dotenv

def get_api_credentials() -> None:
    load_dotenv()
    
    openai_api_key      = os.getenv("OPENAI_API_KEY", "")
    langsmith_api_key   = os.getenv("LANGSMITH_API_KEY", "")
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
    
    if not openai_api_key and not openai_api_key.startswith("sk-"):
        raise ValueError("OPENAI_API_KEY is not set in the environment variables")
    if not langsmith_api_key and not langsmith_api_key.startswith("lsv2-"):
        raise ValueError("LANGSMITH_API_KEY is not set in the environment variables")
    if not huggingface_api_key and not huggingface_api_key.startswith("hf_"):
        raise ValueError("HUGGINGFACE_API_KEY is not set in the environment variables")
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["HUGGINGFACE_API_KEY"] = huggingface_api_key
