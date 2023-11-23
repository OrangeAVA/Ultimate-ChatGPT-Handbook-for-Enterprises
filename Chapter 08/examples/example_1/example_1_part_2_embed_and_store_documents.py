from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import os


def embed_and_store_documents():
    loader = DirectoryLoader("civil_cases", glob="*.docx")
    documents = loader.load()
    embeddings_model = OpenAIEmbeddings()

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
    )
    index_name = "legal-documents-search"
    Pinecone.from_documents(
        documents, embeddings_model, index_name=index_name
    )


if __name__ == "__main__":
    embed_and_store_documents()
