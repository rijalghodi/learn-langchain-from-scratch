"""
Translator

In this script, we create a CLI app that receive an input user 
to translete into language user chose

This project covered concepts:
1. Prompt Template
2. Model
3. Parser
4. Chaining
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = ChatOpenAI(model="gpt-4o")


# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# ----- user input -----

print("💬 Welcome to machine translation app 💬")
language = input("Enter your target language : ")
text = input("Enter your text            : ")
print("Processing...")
print(f"Translation: {chain.invoke({"text": text, "language": language})}")