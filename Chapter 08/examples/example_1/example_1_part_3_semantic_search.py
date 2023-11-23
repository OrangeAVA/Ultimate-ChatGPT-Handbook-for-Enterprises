from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

import pinecone
import os


def retrieve_relevant_cases():
    embeddings_model = OpenAIEmbeddings()
    index_name = "legal-documents-search"
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
    )
    index_name = "legal-documents-search"

    docsearch = Pinecone.from_existing_index(index_name, embeddings_model)
    query = "Cases about financial loss"
    docs = docsearch.similarity_search(query)
    sources = [doc.metadata["source"] for doc in docs]

    return sources


if __name__ == "__main__":
    print(retrieve_relevant_cases())