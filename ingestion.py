from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from const import DIR_PATH, INDEX_NAME


def ingest_docs():
    loader = DirectoryLoader(DIR_PATH)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)

    embeddings = GPT4AllEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(INDEX_NAME)


if __name__ == "__main__":
    ingest_docs()
