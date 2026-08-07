import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import PINECONE_API_KEY

# set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

#define embeding models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = "rag-index"
# retriever function
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever"""
    # ensure the index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new index: {INDEX_NAME}...")
        pc.create_index(
            INDEX_NAME, 
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Created pinecone index")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()
# upload documents to vectorstore
def add_document_to_vectorstore(text_content:str):
    """"
    Adds a single text document to the Pinecone vector store. 
    Splits the text into chunks before embedding and upserting
    """
    if not text_content:
        raise ValueError("Document content cannot be empty!")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True)

    # create langchain document objects from the raw text
    documents= text_splitter.create_documents([text_content])
    print("Splitting document into chunks for indexing")

    # get vectorstore instance to add documents
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    # add documents to vectorstore
    vectorstore.add_documents(documents)
    print("Successfully added chunks to Pinecone vector store")