from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

import pinecone
import os

def embed_and_store_documents():

    loader = PyPDFLoader("Manual_MT655333.pdf")
    manual = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=30,
        length_function=len,
        add_start_index=True,
    )

    documents = text_splitter.split_documents(documents=manual)

    embeddings_model = OpenAIEmbeddings()
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
    )

    index_name = "mt655333-manual"
    Pinecone.from_documents(
        documents, embeddings_model, index_name=index_name
    )


if __name__ == "__main__":
    embed_and_store_documents()
