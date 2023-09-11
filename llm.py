from typing import Any, Dict, List

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS

from const import INDEX_NAME


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = GPT4AllEmbeddings()
    docsearch = FAISS.load_local(
        folder_path= INDEX_NAME,
        embeddings=embeddings,
        index_name="index"
    )
    chat = ChatOllama(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})
