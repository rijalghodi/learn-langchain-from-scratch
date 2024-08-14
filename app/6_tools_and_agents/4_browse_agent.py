from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import datetime
from wikipedia import summary

"""
Browse Agent

This script creates an agent that can be used
"""

# Load environment variables from .env file
load_dotenv()


# Define Tools
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."


# Define the tools that the agent can use
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
]

# Initialize a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

memory = MemorySaver()

agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "0"}}

# Chat Loop to interact with the user
while True:
    user_input = input("Question: ")
    if user_input.lower() == "exit":
        break
    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    print("Answer:", response["messages"][-1].content)