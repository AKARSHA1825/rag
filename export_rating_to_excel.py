"""
export_rating_to_excel.py
------------------------------------------------------------------
NEW entry point for the "product -> styled Excel rating spec" feature.

This does NOT modify app.py, so it will never conflict with your
teammate's changes to the chat CLI. Run it separately:

    python export_rating_to_excel.py

It will:
  1. Ask which product to extract (or take it as a CLI arg).
  2. Reuse the already-initialized Claude client + hybrid retriever
     from rag.py (so embeddings / vector DB are only loaded once).
  3. Gather the FULL context for that product, extract it as
     structured JSON, and build the 2-sheet Excel workbook in the
     same format as the sample file.
  4. Save it under ./output/ and print the path.

FLASK NOTE (for later, if you want an in-browser "Download" button):
Instead of wb.save(path), you can do:

    from io import BytesIO
    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                      download_name=f"{product_name}_RatingSpecification.xlsx",
                      mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

That's a drop-in change inside a Flask route -- nothing here needs to change.
------------------------------------------------------------------
"""

import os
import re
import sys
from datetime import date

# Reuse the client + retriever singletons already created in rag.py
# instead of re-initializing them here (keeps startup fast, avoids
# duplicating config/auth logic, and keeps this file independent of
# rag.py's internals).
from rag import client, retriever
from product_rating_extractor import get_product_rating_data
from rating_excel_builder import build_rating_workbook, save_workbook

OUTPUT_DIR = "output"


def _safe_filename(product_name: str) -> str:
    """Turns a product name into a filesystem-safe filename."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", product_name).strip("_")
    today = date.today().strftime("%d%b%y")
    return f"{slug}_RatingSpecification_{today}.xlsx"


def export_product(product_name: str, debug: bool = False) -> str:
    """
    Runs the full pipeline for one product and returns the saved file path.
    Also usable directly from a Flask route: call this, then send the
    returned path with send_file(), or use the BytesIO variant noted above.
    """
    print(f"Extracting full rating specification for: {product_name}")
    data = get_product_rating_data(client, retriever, product_name, debug=debug)

    n_steps = len(data.get("steps", []))
    n_tables = len(data.get("tables", []))
    print(f"Extracted {n_steps} step section(s) and {n_tables} lookup table(s).")

    wb = build_rating_workbook(product_name, data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, _safe_filename(product_name))
    save_workbook(wb, output_path)

    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    if len(sys.argv) > 1:
        product_name = " ".join(sys.argv[1:]).strip()
    else:
        product_name = input("Enter the product name to extract (e.g. 'Private Motor - Comprehensive'): ").strip()

    if not product_name:
        print("No product name given, exiting.")
        return

    try:
        export_product(product_name, debug=False)
    except Exception as e:
        print(f"Error while exporting: {e}")


if __name__ == "__main__":
    main()