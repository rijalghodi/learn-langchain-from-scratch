"""
Simple Model

This script will use ChatGPT to answer trivial questions
with help of langchain interface
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Invoke the model with a message
result = model.invoke("What is capital city of Japan?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
