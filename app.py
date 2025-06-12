# app.py
import streamlit as st
from MismatchMD1 import get_symptom_results # This line imports your function from MismatchMD1.py

st.set_page_config(page_title="Symptom Checker", layout="centered")

st.title("👨‍⚕️ Simple Medical Symptom Checker")
st.markdown("""
Welcome to our basic symptom checker! Enter a symptom below and get information from various sources.
""")

# Input field for the symptom
user_symptom = st.text_input("Enter your medical symptom here:", placeholder="e.g., fever, headache, cough")

if user_symptom: # This block runs only when the user types something
    st.subheader(f"Results for: **{user_symptom}**")
    results = get_symptom_results(user_symptom) # Calls the function from MismatchMD1.py

    if results: # Check if results were returned (e.g., if your function returns an empty dict for no match)
        for source, info in results.items():
            st.write(f"**{source}:** {info}")
    else:
        st.warning("No specific results found for this symptom. Please try a different symptom or consult a professional.")

st.markdown("---")
st.info("Disclaimer: This is a simple prototype for educational purposes and should not be used for actual medical advice.")