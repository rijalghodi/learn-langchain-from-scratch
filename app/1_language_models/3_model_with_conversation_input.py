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

conversation = [
    SystemMessage(content="You are helpful assistant"),
    HumanMessage(content="Hi, my name is Jim"),
    AIMessage(content="Hi Jim. How can I assist you today?"),
    HumanMessage(content="What is my name?"),    
]

# Invoke the model with a conversation
result = model.invoke(conversation)

print(result.content)
