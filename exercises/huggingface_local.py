# huggingface_local.py

import os
from credentials.get_api_credentials import get_api_credentials
from huggingface_hub import whoami
from langsmith import Client, traceable
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

get_api_credentials()

user = whoami(token = os.environ["HUGGINGFACE_API_KEY"])
if user["auth"]["type"] == "access_token":
    print("You are authenticated with an access token")

langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Define the path of downloading HuggingFace model/tokenizer
os.environ["HF_HOME"] = "/home/knightchaser/huggingfacecache"

llm = HuggingFacePipeline.from_model_id(
    model_id = "microsoft/Phi-3-mini-4k-instruct",
    task = "text-generation",
    pipeline_kwargs = {
        "max_new_tokens": 512,
        "top_k": 50,
        "temperature": 0.1,
        "do_sample": True,
    },
)

# template is already utilizable within the pipeline.
# template = """<|system|>
# You are a helpful assistant.<|end|>
# <|user|>
# {question}<|end|>
# <|assistant|>"""

@traceable
def ask_question(question: str) -> str:
    if not question:
        raise ValueError("Please provide a valid question")
    response = llm.invoke(question)
    return response

if __name__ == "__main__":
    question = input("Enter your question: ")
    print(f"Answer: {ask_question(question)}")
