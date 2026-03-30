import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()

CONNECTION_STRING = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
COLLECTION_NAME = "pdf_chunks"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = PGVector(
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    embeddings=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def build_prompt(context, question):
    return f"""
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def chat():
    print("Digite sua pergunta (ou 'sair'):\n")

    while True:
        question = input(">> ")

        if question.lower() in ["sair", "exit", "quit"]:
            break

        # 🔍 Busca vetorial
        results = vector_store.similarity_search_with_score(
            question,
            k=10
        )

        results = sorted(results, key=lambda x: x[1])

        context = "\n\n".join([doc.page_content for doc, _ in results])

        prompt = build_prompt(context, question)

        response = llm.invoke(prompt)

        print("\nResposta:\n")
        print(response.content)
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    chat()