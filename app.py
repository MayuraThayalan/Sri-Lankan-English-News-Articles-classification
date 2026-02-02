import streamlit as st
import pickle
import re
import numpy as np

# Load the model
@st.cache_resource
def load_models():
    with open('classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('categories.pkl', 'rb') as f:
        categories = pickle.load(f)
    return clf, vectorizer, categories

# Clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    return text.lower().strip()

# Load models once
try:
    clf, vectorizer, categories = load_models()
    st.session_state.models_loaded = True
except:
    st.error("Model files not found! Run pickle save code first.")
    st.stop()

# Application
st.set_page_config(page_title="News Classifier", page_icon="📰")
st.title("📰 News Article Classifier")

# Input section
st.subheader("📝 Enter News Article")
news_text = st.text_area(
    "Paste article here:", 
    placeholder="Try: 'Lionel Messi scores hat-trick in Barcelona victory...'",
    height=150
)

# Prediction button
if st.button("🚀 Predict Category", type="primary", use_container_width=True):
    if news_text.strip():
        # Process text
        with st.spinner("Classifying..."):
            cleaned = clean_text(news_text)
            X_new = vectorizer.transform([cleaned])
            
            # Predict
            pred = clf.predict(X_new)[0]
            prob = np.max(clf.predict_proba(X_new)) * 100
            
            # Show results
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"🎯 {categories[pred].title()}")
            with col2:
                st.metric("Confidence", f"{prob:.1f}%")
            
            # All probabilities
            st.subheader("📊 Confidence Scores")
            probs = clf.predict_proba(X_new)[0] * 100
            for i, cat in enumerate(categories):
                st.progress(probs[i]/100)
                st.caption(f"{cat.title()}: {probs[i]:.1f}%")
                
    else:
        st.warning("⚠️ Please enter some text!")

