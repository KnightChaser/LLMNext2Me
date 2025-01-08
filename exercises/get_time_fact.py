# get_time_fact.py

import os
from dotenv import load_dotenv
from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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

output_parser = DatetimeOutputParser()
output_parser.format = "%Y-%m-%d"

template = """
Answer the users question:

Format Instructions: {format_instructions}

Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(
    template,
    partial_variables={
        # Apply output_parser(DatetimeOutputParser) to the format_instructions
        "format_instructions": output_parser.get_format_instructions()
    },
)


def ask_time_fact(question: str) -> str:
    chain = prompt | ChatOpenAI(model="gpt-3.5-turbo") | output_parser
    output = chain.invoke({
        "question": question
    })
    return output.strftime(output_parser.format)

if __name__ == "__main__":
    question = input("Enter a question about the time fact: ")
    print(f"Answer: {ask_time_fact(question)}")
