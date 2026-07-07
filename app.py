"""Interactive CLI for the RAG application."""
#Inthiyaz
from rag import ask


def main() -> None:
    """Run interactive RAG application."""
    print("\nTraditional RAG Ready")
    print("-" * 50)

    while True:
        try:
            query = input("\nQuestion: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            answer = ask(query, debug=False)
            print("\n" + answer + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
