import fitz


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = "data/rag_sample.pdf"

    pages = extract_text_from_pdf(pdf_path)

    for page in pages:
        print(f"\n--- PAGE {page['page']} ---")
        print(page["text"])