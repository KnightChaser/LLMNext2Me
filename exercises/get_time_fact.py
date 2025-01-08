# get_time_fact.py

from credentials.get_api_credentials import get_api_credentials
from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

get_api_credentials()

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
