import streamlit as st
from transformers import pipeline
import pandas as pd
import re

# ----------------------
# SIMPLE SENTENCE SPLITTER (replaces spaCy)
# ----------------------
def split_sentences(text):
    # Split by ., !, ? → keep clean, non-empty parts
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

# ----------------------
# Load emotion classifier
# ----------------------
classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go-emotions",
    top_k=None
)

# ----------------------
# Predict emotion(s)
# ----------------------
def predict_emotion(sentence, threshold=0.1):
    output = classifier(sentence)[0]  # list of dicts
    labels = [item["label"] for item in output if item["score"] >= threshold]
    return labels if labels else ["neutral"]

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

    # drift score
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
# UI
# ----------------------
st.title("🧠 Emotion Drift Analyzer")

text = st.text_area("Enter your text:", height=200)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        sentences, emoji_list, compact_timeline_list, drift_score = compute_emotion_drift(text)
        severity = classify_drift_severity(drift_score)

        # Show compacted timeline
        st.subheader("🎭 Emotion Timeline")
        st.write(" → ".join(compact_timeline_list))

        # Drift score
        st.subheader("📊 Drift Score")
        st.info(f"**{drift_score:.2f}** ({severity})")

