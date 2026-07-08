"""Interactive CLI for the RAG application."""
from rag import ask, client

# ==================================================================
# NEW: Excel rating-export feature (added by Shaik)
# ------------------------------------------------------------------
from export_rating_to_excel import export_product
from intent_classifier import classify_intent


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

            # ======================================================
            # Handles any phrasing ("pull rating for...", "I need the
            # excel for...", "build rating sheet private motor pls", etc.)
            # ======================================================
            intent = classify_intent(client, query)
            if intent["is_export_request"]:
                if not intent["product_name"]:
                    print("\nWhich product should I generate the rating specification for?\n")
                    continue
                product_name = intent["product_name"]
                print(f"\nGenerating rating specification Excel for: {product_name} ...")
                output_path = export_product(product_name, debug=False)
                print(f"Done. Saved to: {output_path}\n")
                continue

            answer = ask(query, debug=False)
            print("\n" + answer + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()