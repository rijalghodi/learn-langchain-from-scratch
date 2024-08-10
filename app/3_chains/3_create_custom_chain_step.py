from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# We can create a `Runnable` function using `RunnableLambda`

multiply = RunnableLambda(lambda x: x[0]*x[1]) # lamdba function only accept one argument
square = RunnableLambda(lambda x: x**2)
tenth = RunnableLambda(lambda x : x / 10)

chain = multiply | square | tenth

result = chain.invoke((5, 2))

print(result) # 10.0

# ----------------------------------------------------------------
# Let's chain common langchain components with custom runnable function

# Load environment variables from .env
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translete the following into {language}."),
        ("human", "{text}"),
    ]
)

parser = StrOutputParser()

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}")

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | parser | uppercase_output | count_words

result = chain.invoke({"language": "Indonesia", "text": "How are you?"})

print(result)
