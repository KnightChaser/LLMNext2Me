# email_parser.py
import os
from credentials.get_api_credentials import get_api_credentials
from pprint import pprint
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr

get_api_credentials()

# Initialize LangSmith client
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

class EmailSummaryStructure(BaseModel):
    sender: str               = Field(..., title="Sender", description="The email sender")
    sender_email: EmailStr    = Field(..., title="Sender Email", description="The email sender's email address")
    recipient: str            = Field(..., title="Recipient", description="The email recipient")
    recipient_email: EmailStr = Field(..., title="Recipient Email", description="The email recipient's email address")
    subject: str              = Field(..., title="Subject", description="The email subject")
    body: str                 = Field(..., title="Body", description="The email body")
    summary: str              = Field(..., title="Summary", description="The email summary")

# Create the LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

# Create the PydanticOutputParser object
parser = PydanticOutputParser(pydantic_object=EmailSummaryStructure)

# Generate the sample
email_contents = """
From: Robert Johnson <robert.johnson@globaltech.com>
To: Emily Davis <emily.davis@globaltech.com>
Subject: Follow-Up on Marketing Strategy

Dear Emily,

I hope this email finds you well. I wanted to follow up on our recent discussion about the 2024 marketing strategy. To ensure we’re aligned before the next quarter, I suggest we schedule a meeting to finalize the campaign roadmap.

Proposed Agenda:
- Review of Q1 2024 performance metrics
- Adjustments to the campaign strategy based on recent insights
- Allocation of the remaining Q2 budget

Proposed Time: Thursday, January 11, 2025, at 2:30 PM

Please let me know if this time works for you or if there’s a better slot in your calendar. I look forward to your feedback and our continued collaboration to meet our goals for the year.

Best regards,
Robert Johnson
Marketing Manager
GlobalTech Solutions
Phone: (555) 123-4567
Email: robert.johnson@globaltech.com
"""

# Create the prompt template
prompt = PromptTemplate.from_template(
"""
You are a helpful assistant. Please answer the following questions.

QUESTION
{question}

EMAIL CONVERSATION
{email_contents}


FORMAT
{format}
""")

# Add partial formatting for format at PydanticOutputParser
prompt = prompt.partial(format=parser.get_format_instructions())

# Create the chain
chain = prompt | llm | parser

# Wrap the chain invocation with a function that is decorated with @traceable
@traceable
def parse_email(question: str, email_contents: str):
    return chain.invoke(
        {
            "email_contents": email_contents,
            "question": question
        }
    )

# Run the decorated function
response = parse_email("Please summarize the important sections of the given email", email_contents)

# Print the structured result
pprint(response.dict())
