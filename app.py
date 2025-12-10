import streamlit as st
from transformers import pipeline
import re

# ----------------------
# SIMPLE SENTENCE SPLITTER
# ----------------------
def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

# ----------------------
# Load emotion classifier (small CPU-friendly model)
# ----------------------
@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1,  # CPU
        top_k=None
    )

classifier = load_classifier()

# ----------------------
# Predict emotion(s)
# ----------------------
def predict_emotion(sentence, threshold=0.1):
    output = classifier(sentence)[0]
    labels = [item["label"] for item in output if item["score"] >= threshold]
    return labels if labels else ["neutral"]

# ----------------------
# Load sentiment classifier (CPU)
# ----------------------
@st.cache_resource
def load_sentiment_classifier():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

sentiment_classifier = load_sentiment_classifier()

# ----------------------
# Predict overall sentiment
# ----------------------
def predict_overall_sentiment(text):
    result = sentiment_classifier(text)[0]  # {'label': 'POSITIVE', 'score': 0.99}
    return result["label"], result["score"]
# ----------------------
# Emoji mapping
# ----------------------
emotion_emojis = {
    "anger": "😡",
    "joy": "😄",
    "sadness": "😢",
    "fear": "😱",
    "love": "❤️",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "🤢"
}

def emotions_to_emoji(labels):
    return ", ".join([f"{l} {emotion_emojis.get(l, '')}" for l in labels])

# ----------------------
# Compact repeated emotions
# ----------------------
def compact_timeline(emoji_list):
    compact = []
    for e in emoji_list:
        if not compact or e != compact[-1]:
            compact.append(e)
    return compact

# ----------------------
# Compute drift score
# ----------------------
def compute_emotion_drift(text):
    sentences = split_sentences(text)
    emotions_list = [predict_emotion(s) for s in sentences]
    emoji_list = [emotions_to_emoji(e) for e in emotions_list]
    compact = compact_timeline(emoji_list)

    if len(compact) <= 1:
        drift_score = 0
    else:
        changes = sum(compact[i] != compact[i-1] for i in range(1, len(compact)))
        drift_score = changes / (len(compact) - 1)

    return sentences, emoji_list, compact, drift_score

# ----------------------
# Drift severity label
# ----------------------
def classify_drift_severity(score):
    if score == 0:
        return "Stable (No Drift)"
    elif score <= 0.3:
        return "Low Drift"
    elif score <= 0.6:
        return "Moderate Drift"
    else:
        return "High Emotional Volatility"

# ----------------------
# Streamlit UI
# ----------------------
st.title("🧠 Emotion Drift Analyzer")

text = st.text_area("Enter your text:", height=200)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        sentences, emoji_list, compact_timeline_list, drift_score = compute_emotion_drift(text)
        severity = classify_drift_severity(drift_score)

        st.subheader("🎭 Emotion Timeline")
        st.write(" → ".join(compact_timeline_list))

        st.subheader("📊 Drift Score")
        st.info(f"**{drift_score:.2f}** ({severity})")
        # Overall sentiment
        label, score = predict_overall_sentiment(text)
        st.subheader("Overall Sentiment")
        st.info(f"**{label}**")






