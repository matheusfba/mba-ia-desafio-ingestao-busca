from search import search_prompt

def chat():
    print("Digite sua pergunta (ou 'sair'):\n")

    while True:
        question = input(">> ")

        if question.lower() in ["sair", "exit", "quit"]:
            break

        response = search_prompt(question)

        print("\nResposta:\n")
        print(response)
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    chat()
    