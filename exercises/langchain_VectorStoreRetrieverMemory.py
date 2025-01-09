# langchain_VectorStoreRetrieverMemory.py

from credentials.get_api_credentials import get_api_credentials
import faiss
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory

# Get API credentials
get_api_credentials()

# Define the embedding model
embedding_model = OpenAIEmbeddings()

# Initialize the vector store
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vector_store = FAISS(embedding_model, index, InMemoryDocstore({}), {})

# Create the Vector store retriever and memory
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Create a ConversationChain with vector store memory
conversation = ConversationChain(llm=llm, memory=memory)

# Seed the memory with example contexts
memory.save_context(
    inputs={"human": "Introduce yourself"},
    outputs={"AI": "I am an AI assistant. I am here to help you with your queries."},
)
memory.save_context(
    inputs={"human": "What can you do?"},
    outputs={"AI": "I can assist with answering questions, providing suggestions, and more."},
)
memory.save_context(
    inputs={"human": "What is your favorite language?"},
    outputs={"AI": "I don't have preferences, but Python is very popular and versatile."},
)

# Print instructions for the user
print("Chat with the AI! Type 'exit' to end the conversation.")
print("Memory will be used to provide context-aware responses.")
print("\nExample inputs:\n- What can you do?\n- What is your favorite programming language?\n")

# Chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("\nExiting the chat. Here's the stored memory:")
        for doc_id, doc in vector_store.docstore._dict.items():
            print(f"ID: {doc_id}, Content: {doc.page_content}")
        break

    # Display memory retrieval results before generating response
    retrieved_history = memory.load_memory_variables({"human": user_input}).get("history", "No relevant history found.")
    print("\n[Memory Retrieval Results]")
    print(retrieved_history)

    # Get the AI's response
    response = conversation.predict(input=user_input)

    # Print the AI's response
    print(f"\n[AI's current response]")
    print(f"Response: {response}")

    # Optionally display the updated vector store (for debugging)
    print("\n[Updated Memory Store Content]")
    for doc_id, doc in vector_store.docstore._dict.items():
        print(f"ID: {doc_id}, Content: {doc.page_content}")