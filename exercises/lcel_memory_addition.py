# lcel_memory_addition.py
"""
LCEL Memory Addition Module

This module integrates conversation memory into the LCEL (Latent Conversational Embedding Learning)
model using LangChain. It demonstrates a customized conversation chain that:
    - Retrieves conversation history.
    - Formats a prompt with context and user input.
    - Invokes an LLM to generate a response.
    - Saves the conversation context for future reference.

Type hints are added for clarity.
"""

import os
from typing import Any
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load API credentials and initialize the LangSmith client.
get_api_credentials()
langsmith_client: Client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])


@traceable
class CustomizedConversationChain(Runnable):
    """
    A customized conversation chain that integrates conversation memory into the prompt.

    It chains together the following steps:
        1. Retrieve conversation history from memory.
        2. Assign the retrieved history to a key.
        3. Format the prompt with the chat history and new user input.
        4. Invoke the language model.
        5. Parse the output into a clean string.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        prompt: ChatPromptTemplate,
        memory: ConversationBufferMemory,
        input_key: str = "input",
    ) -> None:
        """
        Initialize the conversation chain.

        :param llm: The language model instance.
        :param prompt: The prompt template including system instructions, chat history placeholder, and user input.
        :param memory: The memory instance for storing conversation history.
        :param input_key: The key under which the user's input is passed in the chain.
        """
        self.llm: ChatOpenAI = llm
        self.prompt: ChatPromptTemplate = prompt
        self.memory: ConversationBufferMemory = memory
        self.input_key: str = input_key

        # Build the chain:
        # a. Retrieve chat history from memory using load_memory_variables.
        # b. Extract the value corresponding to memory.memory_key.
        # c. Assign this history under the key "chat_history" so the prompt can access it.
        # d. Format the prompt, call the LLM, and parse the output.
        self.chain: Runnable = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(self.memory.memory_key)
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke_chain(self, query: str) -> str:
        """
        Invoke the conversation chain with the given query.

        :param query: The user's input string.
        :return: The language model's response as a string.
        """
        # Run the chain using the provided query.
        answer: str = self.chain.invoke({self.input_key: query})
        # Save the conversation context: the human's input and the AI's response.
        self.memory.save_context(inputs={"human": query}, outputs={"ai": answer})
        return answer

    def get_chat_history(self) -> Any:
        """
        Retrieve the current conversation history from memory.

        :return: The conversation history stored in memory.
        """
        return self.memory.load_memory_variables({})

    def invoke(self, query: str, **kwargs) -> str:  # type: ignore
        """
        Implement the abstract `invoke` method required by Runnable.

        :param query: The user's input string.
        :return: The language model's response as a string.
        """
        return self.invoke_chain(query)


def main() -> None:
    """
    Main function to initialize components and run an example conversation.
    """
    # Initialize the language model with deterministic output (temperature=0).
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # Create a prompt template with:
    # - A system instruction.
    # - A placeholder for previous chat history.
    # - A human message template for new input.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Initialize conversation memory to store the chat history.
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
    )

    # Create an instance of the customized conversation chain.
    conversation_chain = CustomizedConversationChain(llm, prompt, memory)

    while True:
        user_query = input("You: ")
        response = conversation_chain.invoke_chain(user_query)
        print("AI:", response)
        print("Chat History:", conversation_chain.get_chat_history(), "\n")


if __name__ == "__main__":
    main()

# you: Hello, I'm Daniel. Who are you?
# AI: Hello Daniel, I'm a helpful chatbot here to assist you with any questions or information you may need. How can I help you today?
# Chat History: {'chat_history': [HumanMessage(content="Hello, I'm Daniel. Who are you?", additional_kwargs={}, response_metadata={}), AIMessage(content="Hello Daniel, I'm a helpful chatbot here to assist you with any questions or information you may need. How can I help you today?", additional_kwargs={}, response_metadata={})]}
#
# You: I wonder when the first concept of AI(Artifical Intelligence) came out to the world.
# AI: The concept of artificial intelligence (AI) dates back to ancient times, but the term "artificial intelligence" was coined in 1956 by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon during the Dartmouth Conference. This conference is considered the birth of AI as a field of study. Since then, AI has evolved significantly, with advancements in technology and research leading to the development of various AI applications and technologies.
# Chat History: {'chat_history': [HumanMessage(content="Hello, I'm Daniel. Who are you?", additional_kwargs={}, response_metadata={}), AIMessage(content="Hello Daniel, I'm a helpful chatbot here to assist you with any questions or information you may need. How can I help you today?", additional_kwargs={}, response_metadata={}), HumanMessage(content='I wonder when the first concept of AI(Artifical Intelligence) came out to the world.', additional_kwargs={}, response_metadata={}), AIMessage(content='The concept of artificial intelligence (AI) dates back to ancient times, but the term "artificial intelligence" was coined in 1956 by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon during the Dartmouth Conference. This conference is considered the birth of AI as a field of study. Since then, AI has evolved significantly, with advancements in technology and research leading to the development of various AI applications and technologies.', additional_kwargs={}, response_metadata={})]}
