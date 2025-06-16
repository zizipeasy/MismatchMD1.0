# MismatchMD1.py

# %% (You can keep this if you want, it's a Jupyter/VS Code cell marker)
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time
import json
import requests

class MedicalMismatchDetector:

    # NOTE: These methods are currently defined as instance methods but are called as static methods.
    # For a simple script, we can treat them as helper functions or convert them to staticmethods.
    # For now, I'll adjust the calling convention.

    @staticmethod # Add this decorator to make it a static method, so you don't need an instance of the class
    def deduplicate_results(results):
        seen_urls = set()
        seen_titles = set()
        unique_results = []

        for result in results:
            # Extract title
            title_start = result.find("[From: ") + 7
            title_end = result.find("]", title_start)
            # Handle cases where title might not be found or format is different
            title = result[title_start:title_end].strip() if title_start != -1 and title_end != -1 else ""

            # Extract URLs
            url_start = result.find("[URL: ") + 6
            url_end = result.find("]", url_start)
            # Handle cases where URL might not be found or format is different
            url = result[url_start:url_end].strip() if url_start != -1 and url_end != -1 else ""

            # Add result if both title and URL is unique
            # Ensure title and URL are not empty for uniqueness check
            if url and title and url not in seen_urls and title not in seen_titles:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_results.append(result)
            elif url and not title and url not in seen_urls: # If no title, just check URL
                 seen_urls.add(url)
                 unique_results.append(result)
            elif title and not url and title not in seen_titles: # If no URL, just check title
                 seen_titles.add(title)
                 unique_results.append(result)


        return unique_results

    # Remove this function as Streamlit handles user input
    # def get_user_input():
    #     print("Medical Mismatch Detector")
    #     print("-------------------------")
    #     user_symptoms = input("Enter your symptoms (comma separated): ")
    #     print("\nAnalyzing your symptoms...\n")
    #     return user_symptoms

    # Moved get_diagnosis_from_google inside the class
    @staticmethod # Add this decorator
    def get_diagnosis_from_google(symptoms):
        self_google_diagnoses = []

        API_KEY = "AIzaSyDUGoNMJu0LGUC-L5v8xeToYnSxrRWMIcA"
        SEARCH_ENGINE_ID = "75674d359f1344d0d"
        query = f"self {symptoms} diagnosis"

        try:
            for start in range(1, 31, 10):  # Fetch 20 results in batches of 10
                url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"

                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                for item in data.get("items", []):
                    title = item.get("title", "Untitled")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")

                    if "diagnosis" in snippet.lower() or "condition" in snippet.lower():
                        formatted_text = f"[From: {title}]\n{snippet[:500]}...\n[URL: {link}]"
                        self_google_diagnoses.append(formatted_text)

                time.sleep(1)

        except Exception as e:
            pass  # Suppress errors in demo

        unique_results = MedicalMismatchDetector.deduplicate_results(self_google_diagnoses)
        return unique_results[:5]


    # Moved get_diagnosis_from_pubmed inside the class
    @staticmethod # Add this decorator
    def get_diagnosis_from_pubmed(symptoms):
        self_pubmed_diagnoses = []
        try:
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            # Corrected typo: 'slef' to 'self'
            params = {
                'db': 'pubmed',
                'term': f"self {symptoms} diagnosis", # Corrected 'slef' to 'self'
                'retmode': 'json',
                'retmax': 20
            }

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            id_list = data['esearchresult'].get('idlist', [])
            if not id_list:
                # print("No PubMed IDs found for the given symptoms.")
                return [] # Return empty if no IDs

            for article_id in id_list[:20]: # Reduced to top 5 for quicker demo
                try:
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    summary_params = {
                        'db': 'pubmed',
                        'id': article_id,
                        'retmode': 'json'
                    }

                    summary_response = requests.get(summary_url, params=summary_params, timeout=10)
                    summary_response.raise_for_status()
                    summary_data = summary_response.json()

                    article = summary_data['result'].get(article_id)
                    if article:
                        title = article.get('title', 'No title available')
                        authors = ', '.join(author['name'] for author in article.get('authors', [])) if article.get('authors') else 'No authors available'
                        journal = article.get('source', 'Unknown journal')

                        formatted_text = (
                            f"[From: {title}]\n"
                            f"Authors: {authors}\n"
                            f"Journal: {journal}\n"
                            f"[URL: https://pubmed.ncbi.nlm.nih.gov/{article_id}/]"
                        )
                        self_pubmed_diagnoses.append(formatted_text)

                except requests.exceptions.RequestException as req_e:
                    # print(f"  Network/HTTP Error processing PMID {article_id}: {req_e}")
                    continue
                except Exception as e:
                    # print(f"  Error processing PMID {article_id}: {str(e)}")
                    continue

        except requests.exceptions.RequestException as req_e:
            # print(f"PubMed base search network/HTTP error: {req_e}")
            pass
        except Exception as e:
            # print(f"PubMed search error: {str(e)}")
            pass
            
        unique_results = MedicalMismatchDetector.deduplicate_results(self_pubmed_diagnoses)
        return unique_results[:5]

        return self_pubmed_diagnoses


# NEW FUNCTION: This is the function app.py will call
def get_symptom_results(user_symptoms):
    all_results = {}

    # Get Google results
    # Now call the static method from the class
    google_diagnoses = MedicalMismatchDetector.get_diagnosis_from_google(user_symptoms)
    if google_diagnoses:
        all_results["Google Search Results"] = google_diagnoses
    else:
        all_results["Google Search Results"] = ["No relevant Google results found."]


    # Get PubMed results
    # Now call the static method from the class
    pubmed_diagnoses = MedicalMismatchDetector.get_diagnosis_from_pubmed(user_symptoms)
    if pubmed_diagnoses:
        all_results["Medical Source (PubMed) Results"] = pubmed_diagnoses
    else:
        all_results["Medical Source (PubMed) Results"] = ["No relevant PubMed results found."]

    return all_results

# Modified __main__ block for local testing
if __name__ == "__main__":
    print("Running MismatchMD1.py directly for testing:")
    test_symptoms = input("Enter test symptoms (e.g., 'fever, cough'): ")
    results = get_symptom_results(test_symptoms) # Call the new entry point function

    print("\n--- Combined Results ---")
    for source, info_list in results.items():
        print(f"\n** {source}: **")
        for i, item in enumerate(info_list, 1):
            print(f"Result {i}: \n{item}\n")






# -----------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import re

def extract_clean_text(results, source_type="google"):
    clean_texts = []
    for entry in results:
        if source_type == "google":
            match = re.search(r"\[From: (.*?)\](.*?)\[URL:", entry, re.DOTALL)
            if match:
                content = match.group(2).strip().replace('\n', ' ')
                clean_texts.append(content)

        elif source_type == "pubmed":
            match = re.search(r"\[From: (.*?)\]", entry)
            if match:
                content = match.group(1).strip()
                clean_texts.append(content)
    return clean_texts

# Call the function to get the results and define the variables
# You might want to change the test symptoms here
test_symptoms = "fever, cough" # Or get input again if needed
results = get_symptom_results(test_symptoms)

# Extract the lists from the results dictionary
google_diagnoses = results.get("Google Search Results", [])
pubmed_diagnoses = results.get("Medical Source (PubMed) Results", [])


google_texts = extract_clean_text(google_diagnoses, source_type="google")
pubmed_texts = extract_clean_text(pubmed_diagnoses, source_type="pubmed")


# Topic Modelling
def lda_topic_modeling(docs, n_topics=3):
    # Ensure there are documents to process
    if not docs:
        print("No documents to process for topic modeling.")
        return []

    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError as e:
        # Handle case where no words are found after stop word removal
        print(f"Could not perform topic modeling: {e}")
        return []

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_weights in lda.components_:
        # Get top 10 keywords for each topic
        top_keywords = [feature_names[i] for i in topic_weights.argsort()[-10:]]
        topics.append(top_keywords)
    return topics

# Define the missing plotting function
def plot_keywords_bar(topics, source_name):
    """
    Plots bar charts for keywords in each topic.
    """
    if not topics:
        print(f"No topics to plot for {source_name}.")
        return

    print(f"\n--- Top Keywords for {source_name} Topics ---")
    for i, topic in enumerate(topics):
        print(f"Topic {i+1}: {', '.join(topic)}") # Print keywords for inspection
        # Create a simple bar plot for keywords (assuming equal weight for simplicity from LDA output)
        plt.figure(figsize=(10, 4))
        plt.barh(range(len(topic)), [1] * len(topic), tick_label=topic) # Plot keywords with placeholder values
        plt.yticks(rotation=0) # Ensure vertical labels
        plt.title(f"{source_name} - Topic {i+1} Keywords")
        plt.xlabel("Keyword Presence (Placeholder)") # Indicate that bar height isn't actual weight
        plt.tight_layout()
        plt.show()


google_topics = lda_topic_modeling(google_texts)
pubmed_topics = lda_topic_modeling(pubmed_texts)

# Only attempt to plot if there are topics
if google_topics:
    plot_keywords_bar(google_topics, "Google")
if pubmed_topics:
    plot_keywords_bar(pubmed_topics, "PubMed")


# Reliability
def estimate_reliability(results, source_type):
    data = []
    for res in results:
        title_match = re.search(r"\[From: (.*?)\]", res)
        title = title_match.group(1) if title_match else "Unknown"

        authority = 1 if source_type == "Medical Source (PubMed)" else 0.2 # Adjusted source type check
        citation = 1 if "journal" in res.lower() else 0

        data.append({
            'Source': title[:40] + "...",
            'Authority': authority,
            'Citation': citation
        })
    return pd.DataFrame(data)

# Ensure that google_diagnoses and pubmed_diagnoses are lists
google_df = estimate_reliability(google_diagnoses, "Google Search Results") # Pass the full source name
pubmed_df = estimate_reliability(pubmed_diagnoses, "Medical Source (PubMed) Results") # Pass the full source name

# Only attempt to concatenate if dataframes are not empty
if not google_df.empty or not pubmed_df.empty:
    combined_df = pd.concat([google_df, pubmed_df]).set_index('Source')

    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_df, annot=True, cmap="YlGnBu", cbar=True)
    plt.title("Reliability Assessment Heatmap")
    plt.tight_layout()
    plt.show()
else:
    print("No data available to generate reliability heatmap.")
