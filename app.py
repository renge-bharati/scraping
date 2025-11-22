import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

st.set_page_config(page_title="Blog Title Clustering", page_icon="📝", layout="wide")

st.title("Blog Title Clustering App 📝")

# 1️⃣ Load CSV safely
try:
    df = pd.read_csv("blog_posts_clustered.csv")
except FileNotFoundError:
    st.error("CSV file not found! Make sure 'blog_posts_clustered.csv' is in the root folder.")
    st.stop()

# 2️⃣ Load models safely
try:
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found! Make sure 'kmeans_model.pkl' and 'tfidf_vectorizer.pkl' are in the root folder.")
    st.stop()

# Show scraped data
st.subheader("Scraped & Clustered Blog Titles")
st.dataframe(df[['Title', 'Link', 'Cluster']])

# Predict cluster for new blog title
st.subheader("Predict Cluster for New Blog Title")
new_title = st.text_input("Enter a new blog title:")

if st.button("Predict Cluster"):
    if new_title.strip() != "":
        clean_text = re.sub(r"[^a-zA-Z\s]", "", new_title.lower())
        X_new = vectorizer.transform([clean_text])
        cluster = kmeans.predict(X_new)[0]
        st.success(f"The predicted cluster is: {cluster}")
    else:
        st.error("Please enter a valid blog title.")
