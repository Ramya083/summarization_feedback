import pandas as pd
import nltk
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Download NLTK tokenizer data
nltk.download("punkt")

# Load feedback data
def load_feedback(file_path="feedback.csv"):
    df = pd.read_csv(file_path)
    if "Feedback" not in df.columns:
        raise ValueError("CSV must have a 'Feedback' column.")
    return " ".join(df["Feedback"].dropna().tolist())

# Summarize using HuggingFace BART model
def summarize_bart(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    max_input_length = 1024
    if len(text) > max_input_length:
        text = text[:max_input_length]
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

# Summarize using sumy's TextRank
def summarize_textrank(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# CLI interface
def main():
    text = load_feedback()

    print("Choose summarization method:")
    print("1. BART (Better quality, deep learning)")
    print("2. TextRank (Fast, lightweight)")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        summary = summarize_bart(text)
        method = "BART"
    elif choice == "2":
        summary = summarize_textrank(text)
        method = "TextRank"
    else:
        print("Invalid choice.")
        return

    print(f"\n=== Feedback Summary ({method}) ===\n")
    print(summary)

if __name__ == "__main__":
    main()
