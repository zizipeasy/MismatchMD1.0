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
            for start in range(1, 41, 10):  # Fetch 30 results in batches of 10
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

# Visualization part
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

def extract_clean_text(results, source_type="google"):
    clean_texts = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    for entry in results:
        # Extract URL
        match = re.search(r"\[URL: (.*?)\]", entry)
        url = match.group(1).strip() if match else None

        if url:
            try:
                response = requests.get(url, headers=headers, timeout=8)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract parapraphs
                paragraphs = soup.find_all('p')
                page_text = " ".join(p.get_text() for p in paragraphs)

                # Clean text
                clean_text = re.sub(r'\s+', ' ', page_text.strip())

                if len(clean_text) > 100:
                    clean_texts.append(clean_text)

            except Exception as e:
                continue

    return clean_texts

def lda_topic_modeling(docs, n_topics=3):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_weights in lda.components_:
        top_keywords = [feature_names[i] for i in topic_weights.argsort()[-10:]]
        topics.append(top_keywords)
    return topics

def plot_keywords_bar(topics, label_prefix):
    keywords_flat = [kw for topic in topics for kw in topic]
    keywords_series = pd.Series(keywords_flat)
    top_keywords = keywords_series.value_counts().nlargest(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_keywords.values, y=top_keywords.index, palette="viridis")
    plt.title(f"{label_prefix} Topic Keyword Frequency")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

def estimate_reliability(results, source_type):
    data = []
    for res in results:
        title_match = re.search(r"\[From: (.*?)\]", res)
        title = title_match.group(1) if title_match else "Unknown"

        authority = 1 if source_type.lower() == "pubmed" else 0.2
        citation = 1 if "journal" in res.lower() else 0

        data.append({
            'Source': title[:40] + "...",
            'Authority': authority,
            'Citation': citation
        })
    return pd.DataFrame(data)

def visualize_symptom_analysis(results_dict):
    google_diagnoses = results_dict.get("Google Search Results", [])
    pubmed_diagnoses = results_dict.get("Medical Source (PubMed) Results", [])

    # Skip if placeholder message present
    google_texts = extract_clean_text(google_diagnoses, source_type="google") if "No relevant" not in google_diagnoses[0] else []
    pubmed_texts = extract_clean_text(pubmed_diagnoses, source_type="pubmed") if "No relevant" not in pubmed_diagnoses[0] else []

    if google_texts:
        google_topics = lda_topic_modeling(google_texts)
        plot_keywords_bar(google_topics, "Google")
    if pubmed_texts:
        pubmed_topics = lda_topic_modeling(pubmed_texts)
        plot_keywords_bar(pubmed_topics, "PubMed")

    # Reliability
    google_df = estimate_reliability(google_diagnoses, "google")
    pubmed_df = estimate_reliability(pubmed_diagnoses, "pubmed")
    combined_df = pd.concat([google_df, pubmed_df]).set_index("Source")

    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_df, annot=True, cmap="YlGnBu", cbar=True)
    plt.title("Reliability Assessment Heatmap")
    plt.tight_layout()
    plt.show()

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

    visualize_symptom_analysis(results)
