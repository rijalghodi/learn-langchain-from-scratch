"""
Conversation Input

Model input can be form of list containing conversation
between system, human, and AI as show below
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

conversation = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
conversation.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        break
    conversation.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(conversation)
    response = result.content
    conversation.append(AIMessage(content=response))  # Add AI message

    print(f"\nAI: {response}")


print("\n---- Message History ----\n")
print(conversation)

