from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    create_react_agent,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()


# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


# List of tools available to the agent
tools = [
    Tool(
        name="Time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the current time",
    ),
]

# Initialize a ChatOpenAI model
model = ChatOpenAI(
    model="gpt-4o", temperature=0
)

# Create an agent executor from the agent and tools
agent_executor = create_react_agent(model, tools)

# Run the agent with a test query
response = agent_executor.invoke({"messages": [HumanMessage(content="What time is it?")]})

response = response["messages"][-1].content

# Print the response from the agent
print("response:", response)
