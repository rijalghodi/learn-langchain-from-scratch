import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

###### CREATE VECTOR STORE ######

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
file_path = os.path.join(current_dir, "sources", "romeo_and_juliet.txt")

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to get vector store
def get_vector_store(store_name):
    persistent_directory = os.path.join(db_dir,store_name)
    if not os.path.exists(persistent_directory):
        print("\n --- No vector store ---")
    else:
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        return db

# Function to create and persist vector store
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

# Ensure the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create the vector store and persist it automatically
create_vector_store(docs, "romeo_and_juliet_db")

###### END CREATE VECTOR STORE ######

###### START CREATE CHAIN AND RETRIEVER ######

# Create retriever from vector store
db = get_vector_store("romeo_and_juliet_db")

retriever = db.as_retriever(
    search_type="similarity", # alternatives: mmr, similarity_score_threshold
    search_kwargs={"k": 1},
)

model = ChatOpenAI(model="gpt-3.5-turbo")

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

response = rag_chain.invoke("What is Moby Dick?")

print(response.content)

###### END CHAIN AND RETRIEVER ######


