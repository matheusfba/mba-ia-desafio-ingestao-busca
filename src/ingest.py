import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def batch(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def ingest_pdf():
    pdf_path = "document.pdf"
    connection_string = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
    collection_name = "pdf_chunks"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = PGVector(
        collection_name=collection_name,
        connection=connection_string,
        embeddings=embeddings,
        pre_delete_collection=True
    )

    print(f"Total de chunks: {len(chunks)}")

    BATCH_SIZE = 10
    SLEEP_SECONDS = 5

    for i, chunk_batch in enumerate(batch(chunks, BATCH_SIZE)):
        print(f"Ingerindo batch {i+1}...")

        vector_store.add_documents(chunk_batch)

        print(f"Batch {i+1} concluído. Aguardando...")
        time.sleep(SLEEP_SECONDS)

    print(f"{len(chunks)} chunks armazenados no banco.")


if __name__ == "__main__":
    ingest_pdf()