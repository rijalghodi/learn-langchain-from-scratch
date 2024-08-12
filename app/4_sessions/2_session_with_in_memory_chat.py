#!/usr/bin/env python
from typing import List

from dotenv import load_dotenv
import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

# We will use Conversation class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.conversation import Conversation

load_dotenv()

# create prompt that has two arguments
prompt =  ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant. Reply messages in {language}"),
    ('user', '{query}')
])

model = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt | model| StrOutputParser()


# Session
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

app = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="query") # specify the message key


def ask_with_session(session_id: str, query: str):
    return app.invoke({ "language": "Indonesian", "query": query}, 
               config={"configurable": {"session_id": session_id}})
    
conversation = Conversation(ask_with_session=ask_with_session)

conversation.chat()
