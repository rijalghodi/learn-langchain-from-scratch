#!/usr/bin/env python
from typing import List

from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] =  os.getenv('LANGCHAIN_API_KEY')
os.environ["FIREWORKS_API_KEY"] = os.getenv('FIREWORKS_API_KEY')

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_fireworks import ChatFireworks
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

from langserve import add_routes

# 1. Create prompt template
system_template = "You are a helpful assistant."
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{messages}')
])

# 2. Create model
model = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")


# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = RunnablePassthrough.assign(messages=itemgetter("messages")) | prompt_template | model | parser

# Session
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

app = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")
