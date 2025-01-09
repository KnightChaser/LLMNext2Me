# langchain_ConversationEntityMemory_with_chat.py
# Utilizing conversation entity memory for interactive AI chat.

from credentials.get_api_credentials import get_api_credentials
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# Get API credentials
get_api_credentials()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

memory = ConversationEntityMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=memory,
)

# Print instructions for the user
print("Chat with the AI! Type 'exit' to end the conversation.")
print("Stored memory will be used to maintain context about entities during the conversation.")

# Chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("\nExiting the chat. Here is the final entity memory:")
        print(memory.entity_store.store)
        break
    
    # Get the AI's response
    response = conversation.predict(input=user_input)
    
    # Print the AI's response
    print(f"AI: {response}")

    # Optionally display the memory content for debugging or transparency
    print("\n[Memory Store Content]")
    print(memory.entity_store.store)

