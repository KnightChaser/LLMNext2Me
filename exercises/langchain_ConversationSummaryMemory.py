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

# You: How do bananas grow?
# AI: Bananas grow on plants that are technically classified as herbs, not trees. The banana plant is a perennial herb that grows from an underground stem called a rhizome. The plant grows large, dark green leaves that can reach up to 9 feet in length. The banana fruit itself grows in clusters called hands, with each hand containing multiple individual bananas known as fingers. The fruit starts off as a flower that emerges from the heart of the plant and then develops into a cluster of bananas. The bananas grow upwards towards the sun and are typically harvested when they are still green and then ripened off the plant.
#
# [Current Conversation Summary]
# The human asks how bananas grow. The AI explains that bananas grow on plants classified as herbs, not trees. The banana plant is a perennial herb that grows from an underground stem called a rhizome. The plant produces large, dark green leaves and the fruit grows in clusters called hands, with each hand containing multiple individual bananas known as fingers. The bananas grow upwards towards the sun and are typically harvested when they are still green and then ripened off the plant.
#
# You: How about other fruits such as watermelons?
# AI: Watermelons, unlike bananas, grow on vines that spread out along the ground. The watermelon plant is a member of the Cucurbitaceae family, which also includes cucumbers, pumpkins, and squash. The plant produces large, lobed leaves and yellow flowers that eventually develop into the fruit. The watermelon itself grows on a long, trailing vine and is typically harvested when it reaches full size and sounds hollow when tapped. The fruit is known for its sweet, juicy flesh and is a popular summertime treat.
#
# [Current Conversation Summary]
# The human asks how bananas grow. The AI explains that bananas grow on plants classified as herbs, not trees. The banana plant is a perennial herb that grows from an underground stem called a rhizome. The plant produces large, dark green leaves and the fruit grows in clusters called hands, with each hand containing multiple individual bananas known as fingers. The bananas grow upwards towards the sun and are typically harvested when they are still green and then ripened off the plant. The AI then explains that watermelons, unlike bananas, grow on vines that spread out along the ground. The watermelon plant is a member of the Cucurbitaceae family and produces large, lobed leaves and yellow flowers that develop into the fruit. Watermelons grow on long, trailing vines and are harvested when they reach full size and sound hollow when tapped.
#
# You: ...
