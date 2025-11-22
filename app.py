import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models and data
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

df = pd.read_csv("blog_posts_clustered.csv")

st.title("Blog Title Clustering App 📝")

st.subheader("Scraped Blog Posts")
st.dataframe(df[['Title', 'Link', 'Cluster']])

st.subheader("Predict Cluster for New Blog Title")
new_title = st.text_input("Enter blog title:")

if st.button("Predict Cluster"):
    if new_title.strip() != "":
        clean_text = new_title.lower()
        X_new = vectorizer.transform([clean_text])
        cluster = kmeans.predict(X_new)[0]
        st.success(f"The predicted cluster is: {cluster}")
    else:
        st.error("Please enter a valid blog title.")
