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
print(
    "Stored memory will be used to maintain context about entities during the conversation."
)

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

# You: Hello, who are you?
# AI: Hello! I am an assistant powered by a large language model trained by OpenAI. I am here to help you with a wide range of tasks and provide information on various topics. How can I assist you today?
#
# [Memory Store Content]
# {}
#
# You: Inside large language model, what are inside?
# AI: Inside a large language model, there are multiple components that work together to process and generate text. Some key components include:
#
# 1. Tokenizer: This component breaks down input text into smaller units called tokens, which are the basic building blocks used by the model to understand and generate text.
#
# 2. Embeddings: These are representations of words or tokens in a high-dimensional space that capture semantic relationships between them. Embeddings help the model understand the meaning of words and how they relate to each other.
#
# 3. Neural Network Layers: These layers process the input tokens and embeddings to learn patterns and relationships in the data. They consist of multiple interconnected nodes that perform mathematical operations to transform the input data.
#
# 4. Attention Mechanism: This component helps the model focus on different parts of the input text when generating output. It allows the model to weigh the importance of different tokens in the input sequence.
#
# 5. Decoder: In models like GPT (Generative Pre-trained Transformer), the decoder is responsible for generating the output text based on the processed input. It uses the learned patterns and relationships to predict the next token in the sequence.
#
# Overall, these components work together in a large language model to understand and generate human-like text based on the input it receives.
#
# [Memory Store Content]
# {'OpenAI': 'OpenAI is the organization behind the development of large language models like GPT, which consist of components such as tokenizer, embeddings, neural network layers, attention mechanism, and decoder.'}
#
# You: Are there other kinds of Language Models which are similar to gpt-3.5-turbo?
# AI: Yes, there are other language models that are similar to GPT-3.5-turbo in terms of being large-scale, powerful, and capable of generating human-like text. Some examples of such models include BERT (Bidirectional Encoder Representations from Transformers), XLNet, T5 (Text-to-Text Transfer Transformer), and RoBERTa. These models are also based on transformer architecture and have been trained on large amounts of text data to perform a wide range of natural language processing tasks. Each of these models has its own strengths and weaknesses, and they have been used in various applications such as text generation, language understanding, and machine translation.
#
# [Memory Store Content]
# {'OpenAI': 'OpenAI is the organization behind the development of large language models like GPT, which consist of components such as tokenizer, embeddings, neural network layers, attention mechanism, and decoder.'}
#
# You: ...
