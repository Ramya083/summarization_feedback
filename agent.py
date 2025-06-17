import streamlit as st
import pandas as pd
from transformers import pipeline

# Load local sentiment model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Agent-like logic to analyze and summarize
def analyze_sentiment_agent(series):
    model = load_sentiment_model()
    texts = series.dropna().astype(str).tolist()
    if not texts:
        return None

    results = model(texts)
    total = len(results)
    pos = sum(1 for r in results if r['label'] == 'POSITIVE')
    neg = total - pos  # Only POSITIVE and NEGATIVE labels expected

    pos_pct = round((pos / total) * 100, 1)
    neg_pct = round((neg / total) * 100, 1)

    if pos_pct > 75:
        summary = "Most of the feedback is very positive."
    elif pos_pct > 50:
        summary = "Majority of the feedback is positive."
    elif neg_pct > 75:
        summary = "The feedback is strongly negative."
    elif neg_pct > 50:
        summary = "Most of the feedback is negative."
    else:
        summary = "Feedback is mixed."

    detailed = f"Out of {total} responses, {pos_pct}% are positive and {neg_pct}% are negative. {summary}"
    return {
        "Positive": f"{pos_pct}%",
        "Negative": f"{neg_pct}%",
        "Summary": detailed
    }

# --- Streamlit UI ---
st.set_page_config(page_title="📊 Local Sentiment Analyzer", layout="wide")
st.title("📋 Sentiment Analyzer (No API)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_columns = df.select_dtypes(include='object').columns.tolist()

    st.subheader("📊 Sentiment Summary Table")
    summary_data = []

    for col in text_columns:
        result = analyze_sentiment_agent(df[col])
        if result:
            summary_data.append({
                "Column": col,
                "Positive": result["Positive"],
                "Negative": result["Negative"],
                "Summary": result["Summary"]
            })

    if summary_data:
        st.table(pd.DataFrame(summary_data))
    else:
        st.warning("No valid text columns found to analyze.")
