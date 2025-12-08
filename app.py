import streamlit as st
from transformers import pipeline
import spacy


nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=-1,  # CPU
    top_k=None  # get all scores
)

# Map emotions to emojis
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

def predict_emotion(sentence, threshold=0.1):
    output = classifier(sentence)  
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
        output = output[0] 
    
    labels = [item["label"] for item in output if item["score"] >= threshold]
    return labels if labels else ["unknown"]



def compact_emoji_timeline(emotions_list):
    timeline = []
    for emo_group in emotions_list:
        for e in emo_group:
            entry = f"{emotion_emojis.get(e.lower(), '❓')} {e.lower()}"
            if not timeline or timeline[-1] != entry:
                timeline.append(entry)
    return " → ".join(timeline)


def compute_emotion_drift(text):
    sentences = split_sentences(text)
    emotions_list = [predict_emotion(s) for s in sentences]
    compact_timeline = compact_emoji_timeline(emotions_list)

    flat_emotions = [e[0].lower() if e else "unknown" for e in emotions_list]
    changes = sum(1 for i in range(1, len(flat_emotions)) if flat_emotions[i] != flat_emotions[i-1])
    drift_score = changes / (len(flat_emotions) - 1) if len(flat_emotions) > 1 else 0
    return compact_timeline, drift_score


def classify_drift_severity(score):
    if score == 0:
        return "Stable (No Drift)"
    elif score <= 0.3:
        return "Low Drift"
    elif score <= 0.6:
        return "Moderate Drift"
    else:
        return "High Emotional Volatility"


st.title("Emotion Drift Analyzer")

text = st.text_area("Enter your text:", height=200)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        timeline_str, drift_score = compute_emotion_drift(text)
        severity = classify_drift_severity(drift_score)

        st.subheader("🎨 Emotion Timeline")
        st.write(timeline_str)

        st.subheader("📊 Drift Score")
        st.info(f"**{drift_score:.2f}** ({severity})")
