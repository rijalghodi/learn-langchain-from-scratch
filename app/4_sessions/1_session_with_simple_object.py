"""
Conversation Input

Model input can be form of list containing conversation
between system, human, and AI as show below
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")

sessions = {} # Use a list to store messages

def get_session(session_id: int):
    if  session_id not in sessions:  
        sessions[session_id] = [system_message]
    return sessions[session_id]

def ask_with_session(session_id, question):
    session = get_session(session_id)
    session.append(HumanMessage(content=question)) 
    response = model.invoke(session)
    session.append(response)
    return response

while True:
    session_id = input("\nEnter Session ID: ")
    if session_id.lower() == "exit":
        print("\nApp ended")
        break
    # Chat loops
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("\nSession ended")
            break

        # Get AI response using history
        response = ask_with_session(session_id, query)

        print(f"\nAI: {response.content}")
