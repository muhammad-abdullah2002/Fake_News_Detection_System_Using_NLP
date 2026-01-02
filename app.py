import streamlit as st
import joblib
import re


@st.cache_resource
def load_model():
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Cleaning function 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# App design
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.markdown("### Paste any news article or headline to check if it's **Real** or **Fake**")
st.markdown("---")

user_input = st.text_area("Enter news text here:", height=200, placeholder="e.g. Donald Trump sends embarrassing message...")

if st.button("🔍 Check News", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing the news..."):
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.success("🟢 **This is REAL NEWS**")
            st.balloons()
        else:
            st.error("🔴 **This is FAKE NEWS**")
            st.warning("Caution: This content appears to be misleading or fabricated.")
    else:
        st.warning("Please enter some text to analyze!")

# Sidebar
st.sidebar.header("📊 Model Info")
st.sidebar.success("**Accuracy: ~99%**")
st.sidebar.info("""
- Algorithm: TF-IDF + PassiveAggressiveClassifier
- Trained on: 45,000+ real & fake news articles
- Fast and accurate detection
""")
st.sidebar.markdown("Made with ❤️ by you!")

# Example news
if st.sidebar.button("Try Fake Example"):
    st.text_area("Enter news text here:", value="Donald Trump sends embarrassing New Year message to his haters and the fake news media", height=200)

if st.sidebar.button("Try Real Example"):
    st.text_area("Enter news text here:", value="U.S. military to accept transgender recruits on Monday as ordered by courts", height=200)