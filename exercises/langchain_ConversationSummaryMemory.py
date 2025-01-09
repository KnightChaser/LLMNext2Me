# langchain_ConversationSummaryMemory_with_chat.py
# Utilizing ConversationSummaryMemory for interactive AI chat, which stores conversation in a summarized format for more efficient memory usage.

from credentials.get_api_credentials import get_api_credentials
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

# Get API credentials
get_api_credentials()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Create a ConversationSummaryMemory object
memory = ConversationSummaryMemory(llm=llm)

# Create a ConversationChain object with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
)

# Print instructions for the user
print("Chat with the AI! Type 'exit' to end the conversation.")
print("Conversation summary will be updated after each interaction.")

# Chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("\nExiting the chat. Here is the final conversation summary:")
        print(memory.buffer)
        break

    # Get the AI's response
    response = conversation.predict(input=user_input)

    # Print the AI's response
    print(f"AI: {response}")

    # Optionally display the conversation summary for debugging or transparency
    print("\n[Current Conversation Summary]")
    print(memory.buffer)

