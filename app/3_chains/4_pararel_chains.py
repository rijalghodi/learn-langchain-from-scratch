from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define pros analysis step
def analyze_sentiment(feedback):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are helpful assistant."),
            (
                "human",
                "Classify the sentiment of this feedback as positive, negative, or neutral.\nFeedback:{feedback}.\nSentiment:",
            ),
        ]
    )
    return pros_template.format_prompt(feedback=feedback)


# Define cons analysis step
def analyze_product_name(feedback):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are helpful assistant."),
            (
                "human",
                "Extract the product that is the object of the following feedback. Reply None if a product is not mentioned.\nFeedback: {feedback}.\nProduct Name:",
            ),
        ]
    )
    return cons_template.format_prompt(feedback=feedback)


# Combine pros and cons into a final review
def combine(sentiment, product):
    return {"sentiment": sentiment, "product": product}


# Simplify branches with LCEL
sentiment_branch_chain = (
    RunnableLambda(lambda x: analyze_sentiment(x)) | model | StrOutputParser()
)

product_branch_chain = (
    RunnableLambda(lambda x: analyze_product_name(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    RunnableParallel(branches={"sentiment": sentiment_branch_chain, "product": product_branch_chain})
    | RunnableLambda(lambda x: combine(x["branches"]["sentiment"], x["branches"]["product"]))
)

# Run the chain
result = chain.invoke("laptop is bad!")

# Output
print(result)
