import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "sources", "sangkuriang.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()

# Define the embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # Update to a valid embedding model if needed


# Function to create and persist vector store
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

####### Thare are many ways to split documents. #######

# 1. Character-based Splitting
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "char_chroma_db")

# 2. Sentence-based Splitting
# Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "sent_chroma_db")

# 3. Token-based Splitting
# Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# Useful for transformer models with strict token limits.
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "token_chroma_db")

# 4. Recursive Character-based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "rec_char_chroma_db")

# 5. Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("\n--- Using Custom Splitting ---")

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # Custom logic for splitting text
        return text.split("\n\n")  # Example: split by paragraphs

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "custom_chroma_db")

################################################################

# Function to query a vector store
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory, embedding_function=embeddings
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# Define the user's question
query = "How did Juliet die?"

# Query each vector store
query_vector_store("char_chroma_db", query)
query_vector_store("sent_chroma_db", query)
query_vector_store("token_chroma_db", query)
query_vector_store("char_chroma_db", query)
query_vector_store("custom_chroma_db", query)
