"""
Alternative Models

Here, we try to use various alternative models to answer questions
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Setup environment variables and messages
load_dotenv()

messages = "What is capital city of France?"


# ---- OpenAI Chat Model Example ----

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-opus-20240229")

result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")


# ---- Fireworks with LLAMA Chat Model Example ----
model = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")

result = model.invoke(messages)
print(f"Answer from LLaMA: {result.content}")