# app.py
import streamlit as st

# Import the necessary functions from MismatchMD1.py.
# get_symptom_results fetches the text-based search results.
# visualize_symptom_analysis generates the Matplotlib figures for visualization.
from MismatchMD1 import get_symptom_results, visualize_symptom_analysis

# --- Streamlit Page Configuration ---
# Set the page title that appears in the browser tab.
# Set the layout to "wide" to utilize more screen space, which is good for displaying plots.
st.set_page_config(page_title="Symptom Checker", layout="wide")

# Display the main title of the application.
st.title("MismatchMD")
# Add a brief introductory markdown text.
st.markdown("""
Welcome to our basic symptom checker! Enter a symptom below and get information from various sources,
along with a visual analysis of the findings.
""")

# Input field for the user to enter their medical symptoms.
# 'placeholder' provides a hint to the user.
user_symptom = st.text_input("Enter your medical symptom here:", placeholder="e.g., fever, headache, cough")

# This block executes only when the user has entered some text in the input field.
if user_symptom:
    # Display a subheader indicating the symptoms being analyzed.
    st.subheader(f"Results for: **{user_symptom}**")

    # Display a spinner to indicate that the application is processing,
    # as fetching data can take a moment.
    with st.spinner("Analyzing your symptoms and fetching data... This might take a moment."):
        # Call the function from MismatchMD1.py to get the text search results.
        results = get_symptom_results(user_symptom)

    # Check if any results were returned from the search functions.
    if results:
        st.markdown("---")
        st.subheader("📚 Text-Based Search Results")

        # Use Streamlit columns to display Google and PubMed results side-by-side.
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Google Search Results")
            # Get Google results; default to an empty list if not found.
            google_results = results.get("Google Search Results", [])
            # Check if actual results exist and not just the "No relevant" message.
            if google_results and "No relevant" not in google_results[0]:
                for i, item in enumerate(google_results, 1):
                    st.write(f"**Result {i}:**")
                    # Use st.markdown to render the formatted text, which includes newlines and URLs.
                    st.markdown(item)
                    st.markdown("---") # Add a separator after each result.
            else:
                st.info("No relevant Google results found.")

        with col2:
            st.markdown("#### Medical Source (PubMed) Results")
            # Get PubMed results; default to an empty list if not found.
            pubmed_results = results.get("Medical Source (PubMed) Results", [])
            # Check if actual results exist.
            if pubmed_results and "No relevant" not in pubmed_results[0]:
                for i, item in enumerate(pubmed_results, 1):
                    st.write(f"**Result {i}:**")
                    st.markdown(item)
                    st.markdown("---")
            else:
                st.info("No relevant PubMed results found.")

        # Add some vertical spacing before the visualization section.
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Visualization of Findings")
        st.markdown("---")

        # Call the visualization function from MismatchMD1.py.
        # This function returns a dictionary of Matplotlib Figure objects.
        figures = visualize_symptom_analysis(results)

        # Display the generated figures using st.pyplot().
        # Each plot is checked to ensure it exists (is not None) before attempting to display.
        if figures.get("Google Keywords"):
            st.markdown("#### Google Search Topic Keywords")
            st.pyplot(figures["Google Keywords"]) # Display the bar plot for Google keywords.
            st.markdown("---")
        else:
            st.info("Insufficient Google search data for keyword analysis.")

        if figures.get("PubMed Keywords"):
            st.markdown("#### PubMed Medical Article Topic Keywords")
            st.pyplot(figures["PubMed Keywords"]) # Display the bar plot for PubMed keywords.
            st.markdown("---")
        else:
            st.info("Insufficient PubMed data for keyword analysis.")

        if figures.get("Reliability Heatmap"):
            st.markdown("#### Source Reliability Assessment")
            st.pyplot(figures["Reliability Heatmap"]) # Display the heatmap for source reliability.
            st.markdown("---")
        else:
            st.info("Insufficient data to generate reliability assessment heatmap.")

    else:
        # If no results (text or plots) were found, display a warning.
        st.warning("No specific results found for this symptom. Please try a different symptom or consult a professional.")

st.markdown("---")
# Add a disclaimer at the bottom of the page.
st.info("Disclaimer: This is a simple prototype for educational purposes and should not be used for medical advice.")
