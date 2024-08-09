"""
Chains under the Hood

For more detailed explanation, open 0_chain_under_the_hood.ipynb

Here, we'll recreate chain operation without pipe (|) operator
in order to familiar with technology behind
"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

# Under the hood, chain is a `RunnableSequence` that contains `Runnable` objects
# Many langchain component, including models, prompt templates, and output parsers, are
# implemented `Runnable` interface.

# To replicate langchain component, let's create a `Runnable` function using `RunnableLambda`

double = RunnableLambda(lambda x: 2*x)
square = RunnableLambda(lambda x: x**2)
tenth = RunnableLambda(lambda x : x / 10)

#  3 levels of abstraction of chaining

# ---- Level 0 Abstraction ----
# Chaining is a pipe operation

chain = double | square | tenth

result0 = chain.invoke(5)

# ---- Level 1 Abstraction ----
# Chaining is passing previous output to the next component invocation

chain = tenth.invoke(square.invoke(double.invoke(5)))

result1 = chain.invoke(5)

# ---- Level 2 Abstraction ----
# Chaining is a Runnable Sequence that contains `Runnable` objects

chain = RunnableSequence(first=double, middle=[square], last=tenth)
## or you can just pass Runnables in order
# chain = RunnableSequence(double, square, tenth)
result2 = chain.invoke(5)



# ---- Let's implement this abstraction in chat model ----

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

## or you can just pass it RunnableSequence in order
# chain = RunnableSequence(format_prompt, invoke_model, parse_output)

# Run the chain
response = chain.invoke({"topic": "animals", "joke_count": 3})

# Output
print(response)
