#app.py
import streamlit as st

#Pull in the special tools we need from MismatchMD1.py.
#get_symptom_results finds text-based search results.
#visualize_symptom_analysis creates charts and graphs.
from MismatchMD1 import get_symptom_results, visualize_symptom_analysis

#Streamlit Page Configuration
#Set the title that pops up in your browser tab when you visit the page.
#Page layout is "wide" so we have plenty of room for our charts.
st.set_page_config(page_title="Symptom Checker", layout="wide")

#Displaying the main, big title for our application.
st.title("MismatchMD")
#Add a friendly little introduction to welcome our users.
st.markdown("""
Welcome to our basic symptom checker! Enter a symptom below and get information from various sources,
along with a visual analysis of the findings.
""")

#Create a spot for the user to type in their medical symptoms.
#'placeholder' gives them a helpful hint of what to type.
user_symptom = st.text_input("Enter your medical symptom here:", placeholder="e.g., fever, headache, cough")

#This whole section only springs into action once the user has typed something in the box.
if user_symptom:
    #Show a mini-heading to confirm what symptoms we're looking up.
    st.subheader(f"Results for: **{user_symptom}**")

    #Show a spinning message to let the user know we're busy working behind the scenes,
    #because getting all this info can takes some time lol 
    with st.spinner("Analyzing your symptoms and fetching data... This might take a moment."):
        #Asking MismatchMD1.py to go fetch those text search results for us.
        results = get_symptom_results(user_symptom)

    #Now, let's see if it found anything.
    if results:
        st.markdown("---")
        st.subheader("📚Text-Based Search Results")

        #We'll set up two columns side-by-side to neatly show results from Google and PubMed.
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("####Google Search Results")
            #Grab the Google results; if there aren't any, we'll just have an empty list.
            google_results = results.get("Google Search Results", [])
            #Let's check if we actually got some real results, not just a "no relevant" message.
            if google_results and "No relevant" not in google_results[0]:
                for i, item in enumerate(google_results, 1):
                    st.write(f"**Result {i}:**")
                    #We're using st.markdown here so our formatted text, including new lines and web links, looks just right.
                    st.markdown(item)
                    st.markdown("---") #Add a little line to separate each result, making it easy to read.
            else:
                st.info("No relevant Google results found.")

        with col2:
            st.markdown("####Medical Source (PubMed) Results")
            #Fetch the PubMed results; again, an empty list if none are found.
            pubmed_results = results.get("Medical Source (PubMed) Results", [])
            #Double-check that we have actual results here too.
            if pubmed_results and "No relevant" not in pubmed_results[0]:
                for i, item in enumerate(pubmed_results, 1):
                    st.write(f"**Result {i}:**")
                    st.markdown(item)
                    st.markdown("---")
            else:
                st.info("No relevant PubMed results found.")

        #Add a little empty space here to make our charts look good.
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈Visualization of Findings")
        st.markdown("---")

        #Now, let's ask our artist function from MismatchMD1.py to create charts.
        #This function will hand us back a collection of Matplotlib Figure objects.
        figures = visualize_symptom_analysis(results)

        #Showing off our charts using st.pyplot()
        #We'll check each plot to make sure it actually exists before trying to display it.
        if figures.get("Google Keywords"):
            st.markdown("####Google Search Topic Keywords")
            st.pyplot(figures["Google Keywords"]) 
            st.markdown("---")
        else:
            st.info("Insufficient Google search data for keyword analysis.")

        if figures.get("PubMed Keywords"):
            st.markdown("####PubMed Medical Article Topic Keywords")
            st.pyplot(figures["PubMed Keywords"])
            st.markdown("---")
        else:
            st.info("Insufficient PubMed data for keyword analysis.")

        if figures.get("Reliability Heatmap"):
            st.markdown("####Source Reliability Assessment")
            st.pyplot(figures["Reliability Heatmap"]) 
            st.markdown("---")
        else:
            st.info("Insufficient data to generate reliability assessment heatmap.")

    else:
        #If we didn't find any results at all, let's let the user know gently that the website failed.
        st.warning("No specific results found for this symptom. Please try a different symptom or consult a professional.")

st.markdown("---")
#Always good to add a little note at the bottom, just a friendly reminder.
st.info("Disclaimer: This is a simple prototype for educational purposes and should not be used for medical advice.")
