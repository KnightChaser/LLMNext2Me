# huggingface_endpoint
import os
from credentials.get_api_credentials import get_api_credentials
from huggingface_hub import whoami
from langsmith import Client, traceable
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

get_api_credentials()

user = whoami(token = os.environ["HUGGINGFACE_API_KEY"])
if user["auth"]["type"] == "access_token":
    print("You are authenticated with an access token")

langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])


# Let's use this LLM model
repository_id = "microsoft/Phi-3-mini-4k-instruct"

# This template is recommended to define the input and output format of the model
# according to the official documentation which can be found on HuggingFace.
template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(
    model = repository_id,
    max_new_tokens = 512,
    temperature = 0.1,
    huggingfacehub_api_token = os.environ["HUGGINGFACE_API_KEY"],
)
chain = prompt | llm | StrOutputParser()

@traceable
def ask_question(question: str) -> str:
    response = chain.invoke({"question": question})
    return response

if __name__ == "__main__":
    question = input("Enter your question: ")
    print(f"Answer: {ask_question(question)}")
