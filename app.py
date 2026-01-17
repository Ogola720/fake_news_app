import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import requests
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# 1. CONFIGURATION & ASSETS
st.set_page_config(
    page_title="Veritas | AI Fake News Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data (quietly)
nltk.download('stopwords', quiet=True)

# Load Assets (Lottie Animations) with Error Handling
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None

# Updated Animation URLs (More reliable links)
# News Animation
lottie_news = load_lottieurl("https://lottie.host/08432578-158a-495f-9b05-182393630733/O7o3A2C0l8.json")
# Robot/AI Animation
lottie_robot = load_lottieurl("https://lottie.host/685b88c3-4c9b-4357-9d7a-8c9033320392/e5J57s7kG0.json")


# 2. MODEL LOADING & UTILS

@st.cache_resource
def load_model_and_vectorizer():
    # Load the files we exported from Colab
    try:
        with open('fake_news_classifier.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vc_file:
            vectorizer = pickle.load(vc_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please make sure 'fake_news_classifier.pkl' and 'tfidf_vectorizer.pkl' are in the same folder.")
        return None, None

model, vectorizer = load_model_and_vectorizer()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


# 3. EXPLAINABILITY FUNCTION

def explain_prediction(text, vectorizer, model):
    """
    Analyzes the input text and finds which words contributed most to the Fake vs Real decision.
    """
    clean = clean_text(text)
    words = clean.split()
    
    # Get feature names (vocabulary) and model coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0] # Shape: (n_features,)
    
    # Map words in input to their coefficients
    word_impact = {}
    for word in words:
        if word in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[word]
            score = coefs[idx]
            word_impact[word] = score

    # Sort: Negative scores -> Fake Indicators, Positive scores -> Real Indicators
    sorted_impact = sorted(word_impact.items(), key=lambda x: x[1])
    
    return sorted_impact


# 4. SIDEBAR & NAVIGATION

with st.sidebar:
    # Safe display of Lottie (Only show if loaded successfully)
    if lottie_robot:
        st_lottie(lottie_robot, height=150, key="robot")
    else:
        st.write("🤖") # Fallback emoji if animation fails
        
    st.title("Veritas AI")
    st.markdown("---")
    st.markdown("### ⚙️ Control Panel")
    model_type = st.selectbox("Selected Model", ["PassiveAggressive Classifier (93% Acc)", "LSTM (Coming Soon)"])
    st.markdown("### ℹ️ About")
    st.info("This system uses Natural Language Processing (TF-IDF) to analyze linguistic patterns common in misinformation.")


# 5. MAIN INTERFACE

col1, col2 = st.columns([3, 1])
with col1:
    st.title("Fake News Detector")
    st.markdown("Paste a news article below to evaluate its credibility using AI.")

with col2:
    # Safe display of Lottie
    if lottie_news:
        st_lottie(lottie_news, height=100, key="news_anim")
    else:
        st.write("📰")

# Input Area
news_text = st.text_area("✍️ Article Text", height=200, placeholder="Paste the news content here...")

if st.button("🚀 Analyze Credibility", type="primary"):
    if news_text and model:
        with st.spinner('🤖 AI is reading and analyzing patterns...'):
            
            # 1. Preprocess & Predict
            processed_text = clean_text(news_text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text) # 0 = Fake, 1 = Real
            decision_score = model.decision_function(vectorized_text)[0] # Raw score for confidence
            
            # Normalize confidence to 0-100 scale (approximate for SVM/Linear models)
            # Absolute value of decision_function roughly correlates to confidence distance from hyperplane
            confidence = min(abs(decision_score) * 20 + 50, 100) 
            confidence = max(confidence, 50) # Cap floor at 50%
            
            result_label = "REAL NEWS" if prediction[0] == 1 else "FAKE NEWS"
            color_theme = "green" if prediction[0] == 1 else "red"

         
            # 6. RESULTS DASHBOARD
           
            st.markdown("---")
            
            # A. Big Banner Result
            if prediction[0] == 1:
                st.success(f"### ✅ VERDICT: {result_label}")
            else:
                st.error(f"### 🚨 VERDICT: {result_label}")

            # B. Confidence Gauge (Plotly)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                title = {'text': "AI Confidence Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color_theme},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "white"}],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.plotly_chart(fig_gauge, use_container_width=True)

            # 7. EXPLAINABILITY (Why?)
           
            with col_res2:
                st.subheader("🔍 Why did the AI make this decision?")
                impact_data = explain_prediction(news_text, vectorizer, model)
                
                if not impact_data:
                    st.warning("Not enough significant words found to explain the decision.")
                else:
                    # Separate top fake triggers and top real triggers found IN THIS TEXT
                    fake_triggers = [x for x in impact_data if x[1] < 0][:5] # Lowest scores (Fake)
                    real_triggers = [x for x in impact_data if x[1] > 0][-5:] # Highest scores (Real)
                    
                    # Prepare data for plotting
                    words = []
                    scores = []
                    colors = []
                    
                    if prediction[0] == 0: # If Fake, show Fake indicators
                        st.markdown("**Top words flagging this as SUSPICIOUS:**")
                        for w, s in fake_triggers:
                            words.append(w)
                            scores.append(abs(s)) # Make positive for chart length
                            colors.append('#FF4B4B') # Red
                    else: # If Real, show Real indicators
                        st.markdown("**Top words confirming CREDIBILITY:**")
                        for w, s in real_triggers:
                            words.append(w)
                            scores.append(s)
                            colors.append('#2ECC71') # Green

                    # Bar Chart of Top Keywords
                    fig_bar = go.Figure(go.Bar(
                        x=scores,
                        y=words,
                        orientation='h',
                        marker_color=colors
                    ))
                    fig_bar.update_layout(
                        title="Keyword Influence",
                        xaxis_title="Impact Score",
                        yaxis_title="Word",
                        height=300,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning("⚠️ Please enter some text to analyze.")