# MismatchMD1.py

import requests
import time
import json
import re

# Imports for web scraping and HTML parsing
from bs4 import BeautifulSoup

# Imports for data visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class MedicalMismatchDetector:

    @staticmethod
    def deduplicate_results(results):
        """
        Deduplicates a list of search results based on their URL and title
        to ensure uniqueness.
        """
        seen_urls = set()
        seen_titles = set()
        unique_results = []

        for result in results:
            # Extract the title from the result string using a specific format.
            title_start = result.find("[From: ") + 7
            title_end = result.find("]", title_start)
            title = result[title_start:title_end].strip() if title_start != -1 and title_end != -1 else ""

            # Extract the URL from the result string using a specific format.
            url_start = result.find("[URL: ") + 6
            url_end = result.find("]", url_start)
            url = result[url_start:url_end].strip() if url_start != -1 and url_end != -1 else ""

            # Add result if both title and URL are unique.
            # Handles cases where either title or URL might be missing but the other is present.
            if url and title and url not in seen_urls and title not in seen_titles:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_results.append(result)
            elif url and not title and url not in seen_urls:
                 seen_urls.add(url)
                 unique_results.append(result)
            elif title and not url and title not in seen_titles:
                 seen_titles.add(title)
                 unique_results.append(result)

        return unique_results

    @staticmethod
    def get_diagnosis_from_google(symptoms):
        """
        Fetches self-diagnosis information related to given symptoms from Google Custom Search API.
        """
        self_google_diagnoses = []

        # API key and Search Engine ID for Google Custom Search.
        # NOTE: For production, secure storage of API keys (e.g., environment variables) is recommended.
        API_KEY = "AIzaSyBIwXzA6O2MV3ggjUeQE844bQsZ0TOFnzw"
        SEARCH_ENGINE_ID = "75674d359f1344d0d"
        query = f"{symptoms}"

        try:
            # Iterate to fetch multiple pages of results (up to 30 results in batches of 10).
            for start in range(1, 41, 10):
                # Construct the Google Custom Search API URL.
                url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"

                # Send a GET request to the Google API with a timeout.
                response = requests.get(url, timeout=10)
                # Raise an HTTPError for bad responses (4xx or 5xx status codes).
                response.raise_for_status()
                # Parse the JSON response.
                data = response.json()

                # Iterate through each item (search result) in the response.
                for item in data.get("items", []):
                    # Extract title, snippet, and link from the search item.
                    title = item.get("title", "Untitled")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")

                    # Filter results to include only those relevant to "diagnosis" or "condition".
                    if "diagnosis" in snippet.lower() or "condition" in snippet.lower():
                        # Format the extracted information for display.
                        formatted_text = f"[From: {title}]\n{snippet[:500]}...\n[URL: {link}]"
                        self_google_diagnoses.append(formatted_text)

                # Pause briefly to avoid hitting API rate limits.
                time.sleep(1)

        # Catch specific request exceptions (network issues, HTTP errors).
        except requests.exceptions.RequestException:
            # In a production application, you would log this error. For demo, it's suppressed.
            pass
        # Catch any other general exceptions.
        except Exception:
            # Log other errors if needed.
            pass

        # Deduplicate the fetched results and return the top 5 unique entries.
        unique_results = MedicalMismatchDetector.deduplicate_results(self_google_diagnoses)
        return unique_results[:10]

    @staticmethod
    def get_diagnosis_from_pubmed(symptoms):
        """
        Fetches medical article information related to given symptoms from PubMed (via NCBI E-utilities).
        """
        self_pubmed_diagnoses = []
        try:
            # Base URL for PubMed ESearch utility.
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            # Parameters for the PubMed search query.
            params = {
                'db': 'pubmed', # Specify PubMed database.
                'term': f"{symptoms}", # Search term.
                'retmode': 'json', # Request JSON format for response.
                'retmax': 30 # Request up to 30 article IDs.
            }

            # Send a GET request to PubMed ESearch.
            response = requests.get(base_url, params=params, timeout=10)
            # Raise an HTTPError for bad responses.
            response.raise_for_status()
            # Parse the JSON response.
            data = response.json()

            # Extract the list of article IDs.
            id_list = data['esearchresult'].get('idlist', [])
            if not id_list:
                # If no IDs are found, return an empty list.
                return []

            # Iterate through each article ID to get detailed summaries.
            for article_id in id_list[:20]:
                try:
                    # Base URL for PubMed ESummary utility.
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    # Parameters for the PubMed summary request.
                    summary_params = {
                        'db': 'pubmed',
                        'id': article_id, # Specific article ID.
                        'retmode': 'json'
                    }

                    # Send a GET request for the article summary.
                    summary_response = requests.get(summary_url, params=summary_params, timeout=10)
                    summary_response.raise_for_status()
                    summary_data = summary_response.json()

                    # Extract article details.
                    article = summary_data['result'].get(article_id)
                    if article:
                        # Get title, authors, and journal.
                        title = article.get('title', 'No title available')
                        authors = ', '.join(author['name'] for author in article.get('authors', [])) if article.get('authors') else 'No authors available'
                        journal = article.get('source', 'Unknown journal')

                        # Format the extracted information.
                        formatted_text = (
                            f"[From: {title}]\n"
                            f"Authors: {authors}\n"
                            f"Journal: {journal}\n"
                            f"[URL: https://pubmed.ncbi.nlm.nih.gov/{article_id}/]"
                        )
                        self_pubmed_diagnoses.append(formatted_text)

                # Catch exceptions during individual article summary fetching.
                except requests.exceptions.RequestException:
                    continue # Continue to the next article if an error occurs.
                except Exception:
                    continue

        # Catch exceptions during the initial PubMed search.
        except requests.exceptions.RequestException:
            pass
        except Exception:
            pass

        # Deduplicate the fetched results and return the top 5 unique entries.
        unique_results = MedicalMismatchDetector.deduplicate_results(self_pubmed_diagnoses)
        return unique_results[:10]


def get_symptom_results(user_symptoms):
    """
    Main function to orchestrate fetching results from Google and PubMed.
    Returns a dictionary containing results from both sources.
    """
    all_results = {}

    # Fetch results from Google.
    google_diagnoses = MedicalMismatchDetector.get_diagnosis_from_google(user_symptoms)
    if google_diagnoses:
        all_results["Google Search Results"] = google_diagnoses
    else:
        all_results["Google Search Results"] = ["No relevant Google results found."]

    # Fetch results from PubMed.
    pubmed_diagnoses = MedicalMismatchDetector.get_diagnosis_from_pubmed(user_symptoms)
    if pubmed_diagnoses:
        all_results["Medical Source (PubMed) Results"] = pubmed_diagnoses
    else:
        all_results["Medical Source (PubMed) Results"] = ["No relevant PubMed results found."]

    return all_results


def extract_clean_text(results, source_type="google"):
    """
    Extracts and cleans text content from URLs found in the search results
    to be used for topic modeling.
    """
    clean_texts = []
    headers = {
        # Mimic a browser to avoid some website blocking.
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    
    for entry in results:
        # Use regex to find the URL within the result string.
        match = re.search(r"\[URL: (.*?)\]", entry)
        url = match.group(1).strip() if match else None

        if url:
            try:
                # Send a GET request to the URL with a timeout.
                response = requests.get(url, headers=headers, timeout=8)
                # Raise an error for bad HTTP responses.
                response.raise_for_status()
                # Parse the HTML content using BeautifulSoup.
                soup = BeautifulSoup(response.content, "html.parser")

                # Find all paragraph tags and join their text content.
                paragraphs = soup.find_all('p')
                page_text = " ".join(p.get_text() for p in paragraphs)

                # Clean the text: replace multiple whitespace characters with a single space and strip leading/trailing whitespace.
                clean_text = re.sub(r'\s+', ' ', page_text.strip())

                # Only add text if it's substantial (e.g., more than 100 characters).
                if len(clean_text) > 100:
                    clean_texts.append(clean_text)

            # Catch any exception that occurs during the process (e.g., network error, parsing error).
            except Exception:
                continue # Skip to the next URL if an error occurs.
    return clean_texts

def lda_topic_modeling(docs, n_topics=3):
    """
    Performs Latent Dirichlet Allocation (LDA) to discover topics within a collection of documents.
    """
    # Initialize CountVectorizer to convert text documents into a matrix of token counts.
    # 'stop_words' removes common English words, 'max_df' ignores words appearing in too many documents,
    # 'min_df' ignores words appearing in too few documents.
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    # Fit the vectorizer to the documents and transform them into a document-term matrix.
    X = vectorizer.fit_transform(docs)

    # Initialize Latent Dirichlet Allocation model.
    # 'n_components' is the number of topics to discover.
    # 'random_state' ensures reproducibility.
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    # Fit the LDA model to the document-term matrix.
    lda.fit(X)
    # Get the names of the features (words) from the vectorizer.
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    # Iterate through each topic's word distribution.
    for topic_weights in lda.components_:
        # Get the top 10 keywords for the current topic based on their weights.
        top_keywords = [feature_names[i] for i in topic_weights.argsort()[-10:]]
        topics.append(top_keywords)
    return topics

def create_keywords_bar_plot(topics, label_prefix):
    """
    Generates a bar plot of the most frequently occurring keywords across identified topics.
    Returns the Matplotlib Figure object.
    """
    # Flatten the list of lists of keywords into a single list.
    keywords_flat = [kw for topic in topics for kw in topic]
    # Convert the flattened list into a Pandas Series for easy counting.
    keywords_series = pd.Series(keywords_flat)
    # Get the 15 most frequent keywords and their counts.
    top_keywords = keywords_series.value_counts().nlargest(15)

    # Create a new Matplotlib figure and an axes object within it.
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create a bar plot using Seaborn on the created axes.
    sns.barplot(x=top_keywords.values, y=top_keywords.index, palette="viridis", ax=ax)
    # Set the title of the plot.
    ax.set_title(f"{label_prefix} Topic Keyword Frequency")
    # Set the label for the x-axis.
    ax.set_xlabel("Frequency")
    # Adjust plot layout to prevent labels from overlapping.
    plt.tight_layout()
    # Return the generated figure object.
    return fig

def estimate_reliability(results, source_type):
    """
    Estimates a simplified 'reliability' score for each search result.
    This is a conceptual estimation for demonstration.
    """
    data = []
    for res in results:
        # Extract the title from the result string.
        title_match = re.search(r"\[From: (.*?)\]", res)
        title = title_match.group(1) if title_match else "Unknown"

        # Assign a higher 'Authority' score to PubMed results compared to general Google results.
        authority = 1 if source_type.lower() == "pubmed" else 0.2
        # Assign a 'Citation' score if the word "journal" appears in the result string.
        citation = 1 if "journal" in res.lower() else 0

        # Append the extracted and calculated data for this result.
        data.append({
            'Source': title[:40] + "...", # Truncate title for better display in heatmap.
            'Authority': authority,
            'Citation': citation
        })
    # Convert the list of dictionaries into a Pandas DataFrame.
    return pd.DataFrame(data)

def create_reliability_heatmap(google_df, pubmed_df):
    """
    Generates a heatmap visualizing the estimated reliability across different sources.
    Returns the Matplotlib Figure object.
    """
    # Concatenate Google and PubMed DataFrames and set 'Source' as the index.
    combined_df = pd.concat([google_df, pubmed_df]).set_index("Source")
    # Create a new Matplotlib figure and an axes object.
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create a heatmap using Seaborn on the created axes.
    # 'annot=True' displays the data values on the heatmap.
    # 'cmap="YlGnBu"' sets the color scheme.
    # 'cbar=True' displays the color bar.
    sns.heatmap(combined_df, annot=True, cmap="YlGnBu", cbar=True, ax=ax)
    # Set the title of the heatmap.
    ax.set_title("Reliability Assessment Heatmap")
    # Adjust plot layout.
    plt.tight_layout()
    # Return the generated figure object.
    return fig

def visualize_symptom_analysis(results_dict):
    """
    Orchestrates the creation of all visualization plots based on search results.
    Returns a dictionary where keys are plot names and values are Matplotlib Figure objects.
    If a plot cannot be generated due to lack of data, its value will be None.
    """
    figures = {} # Dictionary to hold all generated figures.

    # Retrieve Google and PubMed diagnosis lists from the results dictionary.
    google_diagnoses = results_dict.get("Google Search Results", [])
    pubmed_diagnoses = results_dict.get("Medical Source (PubMed) Results", [])

    # Initialize lists to hold clean text extracted from URLs.
    google_texts = []
    # Only attempt to extract text if Google results are present and not just the "No relevant" message.
    if google_diagnoses and "No relevant" not in google_diagnoses[0]:
        google_texts = extract_clean_text(google_diagnoses, source_type="google")

    pubmed_texts = []
    # Only attempt to extract text if PubMed results are present and not just the "No relevant" message.
    if pubmed_diagnoses and "No relevant" not in pubmed_diagnoses[0]:
        pubmed_texts = extract_clean_text(pubmed_diagnoses, source_type="pubmed")

    # Generate Google Keywords plot.
    # Only if sufficient text was extracted from Google sources.
    if google_texts:
        google_topics = lda_topic_modeling(google_texts)
        figures["Google Keywords"] = create_keywords_bar_plot(google_topics, "Google")
    else:
        figures["Google Keywords"] = None # Set to None if plot cannot be generated.

    # Generate PubMed Keywords plot.
    # Only if sufficient text was extracted from PubMed sources.
    if pubmed_texts:
        pubmed_topics = lda_topic_modeling(pubmed_texts)
        figures["PubMed Keywords"] = create_keywords_bar_plot(pubmed_topics, "PubMed")
    else:
        figures["PubMed Keywords"] = None # Set to None if plot cannot be generated.

    # Prepare dataframes for reliability assessment.
    google_df = estimate_reliability(google_diagnoses, "google")
    pubmed_df = estimate_reliability(pubmed_diagnoses, "pubmed")

    # Generate Reliability Heatmap.
    # Only if there's any data in either Google or PubMed dataframes for reliability.
    if not google_df.empty or not pubmed_df.empty:
        figures["Reliability Heatmap"] = create_reliability_heatmap(google_df, pubmed_df)
    else:
        figures["Reliability Heatmap"] = None # Set to None if plot cannot be generated.

    return figures # Return the dictionary containing all figure objects (or None).


# This block is for local testing of MismatchMD1.py itself.
# It allows you to run this script directly and see results/plots in separate windows.
if __name__ == "__main__":
    print("Running MismatchMD1.py directly for testing:")
    test_symptoms = input("Enter test symptoms (e.g., 'fever, cough'): ")
    results = get_symptom_results(test_symptoms)

    print("\n--- Combined Text Results ---")
    for source, info_list in results.items():
        print(f"\n** {source}: **")
        for i, item in enumerate(info_list, 1):
            print(f"Result {i}: \n{item}\n")

    print("\n--- Generating Visualizations (may open windows) ---")
    # Get the dictionary of figures from the visualization function.
    figures = visualize_symptom_analysis(results)

    # Loop through the figures and display them in separate Matplotlib windows.
    for title, fig in figures.items():
        if fig:
            print(f"Displaying {title}...")
            # Set the current figure to be displayed.
            plt.figure(fig.number)
            # Show the plot in a new window.
            plt.show()
            # Close the figure to free up memory after displaying.
            plt.close(fig)
        else:
            print(f"No data to display for {title}.")
