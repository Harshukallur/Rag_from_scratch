import re


def clean_text(text):
    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


if __name__ == "__main__":

    sample_text = """
    Machine learning     is a branch of artificial intelligence.
    
    
    It learns patterns from data.
    """

    cleaned = clean_text(sample_text)

    print(cleaned)