import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis model
@st.cache_resource
def get_sentiment_analyzer():
    return pipeline("sentiment-analysis")

# Analyze a column and generate sentiment stats + summary
def analyze_sentiment(series):
    analyzer = get_sentiment_analyzer()
    texts = series.dropna().astype(str).tolist()

    if not texts:
        return None

    results = analyzer(texts)

    positive = sum(1 for r in results if r['label'] == 'POSITIVE')
    negative = sum(1 for r in results if r['label'] == 'NEGATIVE')
    total = len(results)

    pos_pct = round((positive / total) * 100, 1)
    neg_pct = round((negative / total) * 100, 1)

    # Generate detailed summary
    if pos_pct > 75:
        sentiment_summary = "The feedback is overwhelmingly positive."
    elif pos_pct > 50:
        sentiment_summary = "The feedback is mostly positive."
    elif neg_pct > 75:
        sentiment_summary = "The feedback is largely negative."
    elif neg_pct > 50:
        sentiment_summary = "The feedback is mostly negative."
    else:
        sentiment_summary = "The feedback is mixed."

    detailed_summary = (
        f"Out of {total} feedback entries, "
        f"{pos_pct}% were positive and {neg_pct}% were negative. "
        f"{sentiment_summary}"
    )

    return pos_pct, neg_pct, detailed_summary

# Streamlit UI
st.set_page_config(page_title="📊 CSV Sentiment Analyzer", layout="centered")
st.title("📋 Column-wise Sentiment Summary")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_columns = df.select_dtypes(include="object").columns.tolist()

    if not text_columns:
        st.warning("No text columns found in the CSV.")
    else:
        results = []

        for col in text_columns:
            result = analyze_sentiment(df[col])
            if result:
                pos, neg, detail = result
                results.append({
                    "Column": col,
                    "Positive": f"{pos}%",
                    "Negative": f"{neg}%",
                    "Detailed Summary": detail
                })

        if results:
            summary_df = pd.DataFrame(results)
            st.subheader("📊 Sentiment Summary Table")
            st.table(summary_df)
        else:
            st.info("No text content to analyze.")
