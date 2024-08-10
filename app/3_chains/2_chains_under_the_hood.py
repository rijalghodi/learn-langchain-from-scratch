"""
Chains under the Hood

For more detailed explanation, open 0_chain_under_the_hood.ipynb

Here, we'll recreate chain operation without pipe (|) operator
in order to familiar with technology behind
"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

# Under the hood, chain is a `RunnableSequence` that contains `Runnable` objects
# Many langchain component, including models, prompt templates, and output parsers, are
# implemented `Runnable` interface.

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translete the following into {language}."),
        ("human", "{text}"),
    ]
)

parser = StrOutputParser()


#  3 levels of abstraction of chaining

# ---- Level 0 Abstraction ----
# Chaining is a pipe operation

chain = prompt_template | model | parser

result0 = chain.invoke({"language": "Indonesia", "text": "Hi"})

# ---- Level 1 Abstraction ----
# Chaining is passing previous output to the next component invocation

chain = parser.invoke(
    model.invoke(
        prompt_template.invoke({"language": "Indonesia", "text": "Hi"})
        )
    )

result1 = chain.invoke({"language": "Indonesia", "text": "Hi"})

# ---- Level 2 Abstraction ----
# Chaining is a Runnable Sequence that contains `Runnable` objects

chain = RunnableSequence(prompt_template, model, parser)
## or you can just pass Runnables in order
# chain = RunnableSequence(first=prompt_template, middle=[model], last=parser)
result2 = chain.invoke({"language": "Indonesia", "text": "Hi"})

print(result0, result1, result2) # same
