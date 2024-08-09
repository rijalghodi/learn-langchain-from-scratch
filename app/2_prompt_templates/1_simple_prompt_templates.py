"""
Simple Prompt Template

In this script, we'll try to use prompt template to generate ready-to-use prompt
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate

# 1. Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are a great songwriter. Create a song based on the user's request in {language} language:"),
    ('user', "Make a song with the title \"{title}\". Ensure it matches the style of {singer}'s songs.")
])

# ----- user input -----

print("⚒️ Welcome to song prompt factory ⚒️")
language = input("Enter your language : ")
singer = input("Enter your favorite singer : ")
title = input("Enter the song title (or Pick random words): ")
print("Processing...")
print("Here is a song prompt based on your request:")
print(prompt_template.invoke({"language": language, "title": title, "singer": singer}))