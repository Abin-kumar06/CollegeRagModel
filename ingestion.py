import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tempfile import NamedTemporaryFile

def load_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = file
        documents.extend(docs)
    return documents


# ----------- NEW FUNCTION ADDED FOR FRONTEND FILE UPLOAD -----------
def load_uploaded_files(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split(".")[-1]

        with NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        if suffix == "txt":
            loader = TextLoader(tmp_path)
        elif suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            continue

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        documents.extend(docs)

    return documents
# ---------------------------------------------------------------


def main():
    data_dir = 'data'
    persist_dir = 'chroma_db'

    # Load documents
    docs = load_documents(data_dir)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")


if __name__ == "__main__":
    main()