# chat_session_history.py

"""
With using ChatMessageHistory, we can simplify the code by removing the memory instance and the memory_key attribute.
Based on this, we can easily retrieve the chat history from the ChatMessageHistory instance,
enabling us to focus on the specific conversation context and the user's input among multiple conversation contexts.
"""

import os
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load API credentials and initialize the LangSmith client.
get_api_credentials()
langsmith_client: Client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Define the prompt with a MessagesPlaceholder for chat_history.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a question-answering chatbot. Please answer the following questions.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "#Question:\n{question}"),
    ]
)

# Initialize the language model.
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Compose the chain: prompt -> LLM -> output parser.
chain = prompt | llm | StrOutputParser()

# Storage for session histories (each session_id gets its own ChatMessageHistory instance)
llm_session_storage = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a ChatMessageHistory instance for a given session ID."""
    print(f"Conversation session ID: {session_id}")
    if session_id not in llm_session_storage:
        # Create a new ChatMessageHistory instance if this session doesn't exist yet.
        llm_session_storage[session_id] = ChatMessageHistory()
    return llm_session_storage[session_id]


# Create a chain that will automatically fetch and inject chat history.
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",  # The key holding the user's input.
    history_messages_key="chat_history",  # The key in the prompt where chat history is expected.
)


@traceable
def ask_question(session_id: str, question: str) -> str:
    """
    Send a question to the chatbot using a given session ID, and return the chatbot's answer.

    :param session_id: The session identifier for the conversation.
    :param question: The user's question.
    :return: The chatbot's response.
    """
    config = {"configurable": {"session_id": session_id}}
    return chain_with_history.invoke({"question": question}, config=config)


# Example usage
if __name__ == "__main__":
    # Context 1
    print("Context 1:")
    response1 = ask_question("daniel1234", "My name is Daniel.")
    print("AI:", response1)
    response2 = ask_question("daniel1234", "What is my name?")
    print("AI:", response2)

    # Context 2
    print("\nContext 2:")
    response3 = ask_question("unknown1234", "What is my name?")
    print("AI:", response3)

# Context 1:
# Conversation session ID: daniel1234
# AI: Nice to meet you, Daniel! How can I assist you today?
# Conversation session ID: daniel1234
# AI: Your name is Daniel.
#
# Context 2:
# Conversation session ID: unknown1234
# AI: I'm sorry, but I don't have access to your personal information, including your name. How can I assist you today?
