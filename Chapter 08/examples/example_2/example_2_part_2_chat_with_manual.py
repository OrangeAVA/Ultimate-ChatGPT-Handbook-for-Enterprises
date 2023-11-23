from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone

import pinecone
import os

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
)


def retrieve_controllers_info(query, chat_history):
    embeddings_model = OpenAIEmbeddings()
    index_name = "mt655333-manual"
    docsearch = Pinecone.from_existing_index(index_name, embeddings_model)

    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0),
        docsearch.as_retriever(),
        return_source_documents=True,
        condense_question_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":

    chat_history = []
    query = "Does MT65533 support autotuning?"
    
    result = retrieve_controllers_info(query, chat_history)

    print(result["answer"])
    print(result["source_documents"])

    chat_history.append((query, result["answer"]))

    query = "Which methods exactly?"
    result = retrieve_controllers_info(query, chat_history)
    
    print(result["answer"])
    print(result["source_documents"])
