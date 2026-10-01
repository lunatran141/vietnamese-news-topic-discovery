import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Vietnamese News Topic Discovery",
    page_icon="📰",
    layout="wide",
)

st.title("Vietnamese News Topic Discovery")
st.write("Streamlit application skeleton is running.")

st.subheader("API Status")

if st.button("Check API health"):
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        st.success(f"API is healthy: {response.json()}")
    except requests.RequestException as error:
        st.error(f"Cannot connect to API: {error}")