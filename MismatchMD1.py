#MismatchMD1.py

#Imports the requests library to handle HTTP requests for web data.
import requests
#Imports the time library for delaying execution, primarily to manage API request rates.
import time
#Imports the json library for parsing JSON formatted data, common in web APIs.
import json
#Imports the re module for regular expressions, used for pattern matching in strings.
import re

#Imports for web scraping and HTML parsing
#Imports BeautifulSoup for parsing HTML and XML documents, essential for web scraping.
from bs4 import BeautifulSoup

#Imports for data visualization and analysis
#Imports Matplotlib's pyplot module for creating static, interactive, and animated visualizations.
import matplotlib.pyplot as plt
#Imports Seaborn, a data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns
#Imports Pandas, a powerful library for data manipulation and analysis, particularly with DataFrames.
import pandas as pd
#Imports CountVectorizer from scikit-learn for converting text collections into a matrix of token counts.
from sklearn.feature_extraction.text import CountVectorizer
#Imports LatentDirichletAllocation (LDA) from scikit-learn for topic modeling, identifying underlying topics in a collection of documents.
from sklearn.decomposition import LatentDirichletAllocation


#Defines a class to encapsulate functions related to detecting medical information mismatches.
class MedicalMismatchDetector:

    #Decorator indicating that this method belongs to the class but does not operate on instance-specific data.
    @staticmethod
    #Method to remove duplicate entries from a list of search results.
    def deduplicate_results(results):
        """
        Deduplicates a list of search results based on their URL and title
        to ensure uniqueness.
        """
        #Initializes a set to store unique URLs encountered, ensuring quick lookup and uniqueness.
        seen_urls = set()
        #Initializes a set to store unique titles encountered.
        seen_titles = set()
        #Initializes a list to store the deduplicated results.
        unique_results = []

        #Iterates through each search result in the input list.
        for result in results:
            #Extract the title from the result string using a specific format.
            #Finds the starting index of the title after the "[From: " tag.
            title_start = result.find("[From: ") + 7
            #Finds the ending index of the title at the closing bracket.
            title_end = result.find("]", title_start)
            #Extracts the title, stripping whitespace, or sets to empty if not found.
            title = result[title_start:title_end].strip() if title_start != -1 and title_end != -1 else ""

            #Extract the URL from the result string using a specific format.
            #Finds the starting index of the URL after the "[URL: " tag.
            url_start = result.find("[URL: ") + 6
            #Finds the ending index of the URL at the closing bracket.
            url_end = result.find("]", url_start)
            #Extracts the URL, stripping whitespace, or sets to empty if not found.
            url = result[url_start:url_end].strip() if url_start != -1 and url_end != -1 else ""

            #Add result if both title and URL are unique.
            #Handles cases where either title or URL might be missing but the other is present.
            #Checks if both URL and title are present and have not been seen before.
            if url and title and url not in seen_urls and title not in seen_titles:
                #Adds the URL to the set of seen URLs.
                seen_urls.add(url)
                #Adds the title to the set of seen titles.
                seen_titles.add(title)
                #Appends the result to the list of unique results.
                unique_results.append(result)
            #Checks if only the URL is present and unique.
            elif url and not title and url not in seen_urls:
                #Adds the URL to the set of seen URLs.
                seen_urls.add(url)
                #Appends the result to the list of unique results.
                unique_results.append(result)
            #Checks if only the title is present and unique.
            elif title and not url and title not in seen_titles:
                #Adds the title to the set of seen titles.
                seen_titles.add(title)
                #Appends the result to the list of unique results.
                unique_results.append(result)

        #Returns the list containing only unique search results.
        return unique_results

    @staticmethod
    #Method to fetch self-diagnosis information from Google Custom Search based on provided symptoms.
    def get_diagnosis_from_google(symptoms):
        """
        Fetches self-diagnosis information related to given symptoms from Google Custom Search API.
        """
        #Initializes a list to store diagnosis results from Google.
        self_google_diagnoses = []

        #API key and Search Engine ID for Google Custom Search.
        #NOTE: For production, secure storage of API keys (e.g., environment variables) is recommended.
        #Google Custom Search API Key. (Note: Hardcoding API keys is not recommended for production environments.)
        API_KEY = "AIzaSyBIwXzA6O2MV3ggjUeQE844bQsZ0TOFnzw"
        #Google Custom Search Engine ID.
        SEARCH_ENGINE_ID = "75674d359f1344d0d"
        #Constructs the search query string using the provided symptoms.
        query = f"{symptoms}"

        #Begins a try block to handle potential exceptions during API requests.
        try:
            #Iterate to fetch multiple pages of results (up to 30 results in batches of 10).
            #Loops to retrieve results in batches, controlling the 'start' parameter for pagination.
            for start in range(1, 41, 10):
                #Construct the Google Custom Search API URL.
                #Builds the full API request URL.
                url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"

                #Send a GET request to the Google API with a timeout.
                #Sends an HTTP GET request to the Google Custom Search API with a 10-second timeout.
                response = requests.get(url, timeout=10)
                #Raise an HTTPError for bad responses (4xx or 5xx status codes).
                #Raises an HTTPError for unsuccessful status codes (e.g., 404, 500).
                response.raise_for_status()
                #Parse the JSON response.
                #Parses the JSON response body into a Python dictionary.
                data = response.json()

                #Iterate through each item (search result) in the response.
                #Iterates through each search result item within the 'items' key of the JSON response.
                for item in data.get("items", []):
                    #Extract title, snippet, and link from the search item.
                    #Extracts the title of the search result.
                    title = item.get("title", "Untitled")
                    #Extracts the snippet (short description) of the search result.
                    snippet = item.get("snippet", "")
                    #Extracts the URL link of the search result.
                    link = item.get("link", "")

                    #Filter results to include only those relevant to "diagnosis" or "condition".
                    #Filters results where the snippet contains "diagnosis" or "condition" (case-insensitive).
                    if "diagnosis" in snippet.lower() or "condition" in snippet.lower():
                        #Format the extracted information for display.
                        #Formats the extracted title, snippet (truncated to 500 characters), and URL.
                        formatted_text = f"[From: {title}]\n{snippet[:500]}...\n[URL: {link}]"
                        #Adds the formatted text to the list of Google diagnoses.
                        self_google_diagnoses.append(formatted_text)

                #Pause briefly to avoid hitting API rate limits.
                #Pauses execution for 1 second to comply with API rate limits.
                time.sleep(1)

        #Catch specific request exceptions (network issues, HTTP errors).
        #Catches exceptions related to network issues or invalid HTTP responses.
        except requests.exceptions.RequestException:
            #In a production application, you would log this error. For demo, it's suppressed.
            pass #Silently handles the exception (in a real application, this should be logged).
        #Catch any other general exceptions.
        except Exception:
            #Log other errors if needed.
            pass #Silently handles the exception.

        #Deduplicate the fetched results and return the top 5 unique entries.
        #Deduplicates the collected Google diagnosis results.
        unique_results = MedicalMismatchDetector.deduplicate_results(self_google_diagnoses)
        #Returns up to the first 10 unique results.
        return unique_results[:10]

    @staticmethod
    #Method to fetch medical article information from PubMed based on symptoms.
    def get_diagnosis_from_pubmed(symptoms):
        """
        Fetches medical article information related to given symptoms from PubMed (via NCBI E-utilities).
        """
        #Initializes a list to store diagnosis results from PubMed.
        self_pubmed_diagnoses = []
        #Begins a try block to handle potential exceptions during API requests.
        try:
            #Base URL for PubMed ESearch utility.
            #Base URL for NCBI ESearch to find PubMed article IDs.
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            #Parameters for the PubMed search query.
            #Dictionary of parameters for the ESearch API call.
            params = {
                'db': 'pubmed', #Specifies the PubMed database.
                'term': f"{symptoms}", #The search term (symptoms).
                'retmode': 'json', #Requests the response in JSON format.
                'retmax': 30 #Requests a maximum of 30 article IDs.
            }

            #Send a GET request to PubMed ESearch.
            #Sends an HTTP GET request to PubMed ESearch.
            response = requests.get(base_url, params=params, timeout=10)
            #Raise an HTTPError for bad responses.
            response.raise_for_status()
            #Parse the JSON response.
            response.json()
            #Parses the JSON response.
            data = response.json()

            #Extract the list of article IDs.
            #Extracts the list of article IDs from the ESearch result.
            id_list = data['esearchresult'].get('idlist', [])
            #Checks if no article IDs were found.
            if not id_list:
                return [] #Returns an empty list if no IDs are available.

            #Iterate through each article ID to get detailed summaries.
            #Iterates through the first 20 article IDs obtained.
            for article_id in id_list[:20]:
                #Begins an inner try block for fetching individual article summaries.
                try:
                    #Base URL for PubMed ESummary utility.
                    #Base URL for NCBI ESummary to retrieve article summaries.
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    #Parameters for the PubMed summary request.
                    #Dictionary of parameters for the ESummary API call.
                    summary_params = {
                        'db': 'pubmed', #Specifies the PubMed database.
                        'id': article_id, #The specific article ID for which to get the summary.
                        'retmode': 'json' #Requests the response in JSON format.
                    }

                    #Send a GET request for the article summary.
                    #Sends an HTTP GET request for the article summary.
                    summary_response = requests.get(summary_url, params=summary_params, timeout=10)
                    summary_response.raise_for_status()
                    #Parses the JSON response for the summary.
                    summary_data = summary_response.json()

                    #Extract article details.
                    #Extracts the specific article details from the summary data.
                    article = summary_data['result'].get(article_id)
                    #Checks if article details were successfully retrieved.
                    if article:
                        #Get title, authors, and journal.
                        #Extracts the article title.
                        title = article.get('title', 'No title available')
                        #Extracts and formats author names.
                        authors = ', '.join(author['name'] for author in article.get('authors', [])) if article.get('authors') else 'No authors available'
                        #Extracts the journal name.
                        journal = article.get('source', 'Unknown journal')

                        #Format the extracted information.
                        #Formats the extracted article information.
                        formatted_text = (
                            f"[From: {title}]\n"
                            f"Authors: {authors}\n"
                            f"Journal: {journal}\n"
                            f"[URL: https://pubmed.ncbi.nlm.nih.gov/{article_id}/]" #Constructs the PubMed URL for the article.
                        )
                        #Adds the formatted text to the list of PubMed diagnoses.
                        self_pubmed_diagnoses.append(formatted_text)

                #Catch exceptions during individual article summary fetching.
                #Catches request-related exceptions for individual summary fetches.
                except requests.exceptions.RequestException:
                    continue #Continues to the next article ID if an error occurs.
                #Catches other general exceptions for individual summary fetches.
                except Exception:
                    continue #Continues to the next article ID.

        #Catch exceptions during the initial PubMed search.
        #Catches request-related exceptions for the initial PubMed search.
        except requests.exceptions.RequestException:
            pass #Silently handles the exception.
        #Catches other general exceptions for the initial PubMed search.
        except Exception:
            pass #Silently handles the exception.

        #Deduplicate the fetched results and return the top 5 unique entries.
        #Deduplicates the collected PubMed diagnosis results.
        unique_results = MedicalMismatchDetector.deduplicate_results(self_pubmed_diagnoses)
        #Returns up to the first 10 unique results.
        return unique_results[:10]


#Main function to coordinate fetching results from Google and PubMed.
def get_symptom_results(user_symptoms):
    """
    Main function to orchestrate fetching results from Google and PubMed.
    Returns a dictionary containing results from both sources.
    """
    #Initializes an empty dictionary to store results from both sources.
    all_results = {}

    #Fetch results from Google.
    #Calls the method to get diagnosis information from Google.
    google_diagnoses = MedicalMismatchDetector.get_diagnosis_from_google(user_symptoms)
    #Checks if any Google diagnoses were found.
    if google_diagnoses:
        #Stores the Google results in the dictionary.
        all_results["Google Search Results"] = google_diagnoses
    #If no Google diagnoses were found.
    else:
        #Adds a placeholder message indicating no results.
        all_results["Google Search Results"] = ["No relevant Google results found."]

    #Fetch results from PubMed.
    #Calls the method to get diagnosis information from PubMed.
    pubmed_diagnoses = MedicalMismatchDetector.get_diagnosis_from_pubmed(user_symptoms)
    #Checks if any PubMed diagnoses were found.
    if pubmed_diagnoses:
        #Stores the PubMed results in the dictionary.
        all_results["Medical Source (PubMed) Results"] = pubmed_diagnoses
    #If no PubMed diagnoses were found.
    else:
        #Adds a placeholder message indicating no results.
        all_results["Medical Source (PubMed) Results"] = ["No relevant PubMed results found."]

    #Returns the dictionary containing results from both sources.
    return all_results


#Function to extract and clean text content from URLs for topic modeling.
def extract_clean_text(results, source_type="google"):
    """
    Extracts and cleans text content from URLs found in the search results
    to be used for topic modeling.
    """
    #Initializes a list to store the extracted and cleaned text.
    clean_texts = []
    #Defines HTTP headers to mimic a web browser, potentially avoiding blocks.
    #Mimic a browser to avoid some website blocking.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    
    #Iterates through each search result entry.
    for entry in results:
        #Use regex to find the URL within the result string.
        #Uses a regular expression to find the URL embedded in the result string.
        match = re.search(r"\[URL: (.*?)\]", entry)
        #Extracts the URL if a match is found, stripping whitespace.
        url = match.group(1).strip() if match else None

        #Proceeds only if a URL was successfully extracted.
        if url:
            #Begins a try block to handle potential exceptions during web scraping.
            try:
                #Send a GET request to the URL with a timeout.
                #Sends an HTTP GET request to the URL with specified headers and timeout.
                response = requests.get(url, headers=headers, timeout=8)
                #Raise an error for bad HTTP responses.
                response.raise_for_status()
                #Parse the HTML content using BeautifulSoup.
                #Parses the HTML content of the response using BeautifulSoup.
                soup = BeautifulSoup(response.content, "html.parser")

                #Find all paragraph tags and join their text content.
                #Finds all paragraph (`<p>`) tags in the HTML.
                paragraphs = soup.find_all('p')
                #Concatenates the text content of all paragraphs into a single string.
                page_text = " ".join(p.get_text() for p in paragraphs)

                #Clean the text: replace multiple whitespace characters with a single space and strip leading/trailing whitespace.
                #Replaces multiple whitespace characters with a single space and removes leading/trailing whitespace.
                clean_text = re.sub(r'\s+', ' ', page_text.strip())

                #Only add text if it's substantial (e.g., more than 100 characters).
                #Checks if the cleaned text length is substantial (more than 100 characters).
                if len(clean_text) > 100:
                    #Appends the cleaned text to the list.
                    clean_texts.append(clean_text)

            #Catch any exception that occurs during the process (e.g., network error, parsing error).
            #Catches any exception that occurs during the URL request or parsing.
            except Exception:
                continue #Continues to the next URL if an error occurs.
    #Returns the list of extracted and cleaned texts.
    return clean_texts

#Performs Latent Dirichlet Allocation (LDA) for topic discovery in text documents.
def lda_topic_modeling(docs, n_topics=3):
    """
    Performs Latent Dirichlet Allocation (LDA) to discover topics within a collection of documents.
    """
    #Initialize CountVectorizer to convert text documents into a matrix of token counts.
    #'stop_words' removes common English words, 'max_df' ignores words appearing in too many documents,
    #'min_df' ignores words appearing in too few documents.
    #Initializes CountVectorizer to convert text into numerical token counts, applying stop word removal and frequency filtering.
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    #Fit the vectorizer to the documents and transform them into a document-term matrix.
    #Fits the vectorizer to the input documents and transforms them into a document-term matrix.
    X = vectorizer.fit_transform(docs)

    #Initialize Latent Dirichlet Allocation model.
    #'n_components' is the number of topics to discover.
    #'random_state' ensures reproducibility.
    #Initializes the LDA model with a specified number of topics and a random state for reproducibility.
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    #Fit the LDA model to the document-term matrix.
    #Fits the LDA model to the document-term matrix.
    lda.fit(X)
    #Get the names of the features (words) from the vectorizer.
    #Retrieves the actual words (feature names) from the vectorizer.
    feature_names = vectorizer.get_feature_names_out()
    #Initializes a list to store the keywords for each discovered topic.
    topics = []
    #Iterate through each topic's word distribution.
    #Iterates through the word distributions for each topic.
    for topic_weights in lda.components_:
        #Get the top 10 keywords for the current topic based on their weights.
        #Identifies the top 10 keywords for the current topic based on their weights.
        top_keywords = [feature_names[i] for i in topic_weights.argsort()[-10:]]
        #Adds the list of top keywords to the topics list.
        topics.append(top_keywords)
    #Returns the list of topics, each represented by its top keywords.
    return topics

#Generates a bar plot showing the frequency of keywords across identified topics.
def create_keywords_bar_plot(topics, label_prefix):
    """
    Generates a bar plot of the most frequently occurring keywords across identified topics.
    Returns the Matplotlib Figure object.
    """
    #Flatten the list of lists of keywords into a single list.
    #Flattens the list of topic keywords into a single list.
    keywords_flat = [kw for topic in topics for kw in topic]
    #Convert the flattened list into a Pandas Series for easy counting.
    #Converts the flat list of keywords into a Pandas Series.
    keywords_series = pd.Series(keywords_flat)
    #Get the 15 most frequent keywords and their counts.
    #Counts the occurrences of each keyword and selects the top 15 most frequent.
    top_keywords = keywords_series.value_counts().nlargest(15)

    #Create a new Matplotlib figure and an axes object within it.
    #Creates a new Matplotlib figure and a set of subplots with a specified size.
    fig, ax = plt.subplots(figsize=(10, 6))
    #Create a bar plot using Seaborn on the created axes.
    # 'annot=True' displays the data values on the heatmap.
    # 'cmap="YlGnBu"' sets the color scheme.
    # 'cbar=True' displays the color bar.
    #Generates a bar plot using Seaborn, displaying keyword frequencies.
    sns.barplot(x=top_keywords.values, y=top_keywords.index, palette="viridis", ax=ax)
    #Set the title of the plot.
    #Sets the title of the bar plot, including a prefix (e.g., "Google").
    ax.set_title(f"{label_prefix} Topic Keyword Frequency")
    #Set the label for the x-axis.
    #Sets the label for the x-axis.
    ax.set_xlabel("Frequency")
    #Adjust plot layout to prevent labels from overlapping.
    #Adjusts plot parameters for a tight layout, preventing labels from overlapping.
    plt.tight_layout()
    #Return the generated figure object.
    return fig #Returns the generated Matplotlib Figure object.

#Estimates a simplified reliability score for each search result.
def estimate_reliability(results, source_type):
    """
    Estimates a simplified 'reliability' score for each search result.
    This is a conceptual estimation for demonstration.
    """
    #Initializes an empty list to store reliability data for each result.
    data = []
    #Iterates through each search result.
    for res in results:
        #Extract the title from the result string.
        #Uses regex to extract the title from the result string.
        title_match = re.search(r"\[From: (.*?)\]", res)
        #Extracts the title or sets it to "Unknown" if not found.
        title = title_match.group(1) if title_match else "Unknown"

        #Assign a higher 'Authority' score to PubMed results compared to general Google results.
        #Assigns an 'Authority' score: 1 for PubMed, 0.2 for other sources.
        authority = 1 if source_type.lower() == "pubmed" else 0.2
        #Assign a 'Citation' score if the word "journal" appears in the result string.
        #Assigns a 'Citation' score: 1 if "journal" is in the text, 0 otherwise.
        citation = 1 if "journal" in res.lower() else 0

        #Append the extracted and calculated data for this result.
        #Appends a dictionary containing source, authority, and citation scores for the current result.
        data.append({
            'Source': title[:40] + "...", #Truncates the title for display and labels it as 'Source'.
            'Authority': authority, #The calculated authority score.
            'Citation': citation #The calculated citation score.
        })
    #Convert the list of dictionaries into a Pandas DataFrame.
    return pd.DataFrame(data) #Converts the list of dictionaries into a Pandas DataFrame.

#Generates a heatmap to visualize estimated reliability scores.
def create_reliability_heatmap(google_df, pubmed_df):
    """
    Generates a heatmap visualizing the estimated reliability across different sources.
    Returns the Matplotlib Figure object.
    """
    #Concatenate Google and PubMed DataFrames and set 'Source' as the index.
    #Concatenates the Google and PubMed DataFrames and sets the 'Source' column as the index.
    combined_df = pd.concat([google_df, pubmed_df]).set_index("Source")
    #Create a new Matplotlib figure and an axes object.
    #Creates a new Matplotlib figure and an axes object with a specified size.
    fig, ax = plt.subplots(figsize=(10, 6))
    #Create a heatmap using Seaborn on the created axes.
    # 'annot=True' displays the data values on the heatmap.
    # 'cmap="YlGnBu"' sets the color scheme.
    # 'cbar=True' displays the color bar.
    #Generates a heatmap using Seaborn, displaying annotations, a color map, and a color bar.
    sns.heatmap(combined_df, annot=True, cmap="YlGnBu", cbar=True, ax=ax)
    #Set the title of the heatmap.
    #Sets the title of the heatmap.
    ax.set_title("Reliability Assessment Heatmap")
    #Adjust plot layout.
    #Adjusts plot parameters for a tight layout.
    plt.tight_layout()
    #Return the generated figure object.
    return fig #Returns the generated Matplotlib Figure object.

#Orchestrates the creation of various visualization plots based on search results.
def visualize_symptom_analysis(results_dict):
    """
    Orchestrates the creation of all visualization plots based on search results.
    Returns a dictionary where keys are plot names and values are Matplotlib Figure objects.
    If a plot cannot be generated due to lack of data, its value will be None.
    """
    #Initializes an empty dictionary to store generated Matplotlib figure objects.
    figures = {}

    #Retrieve Google and PubMed diagnosis lists from the results dictionary.
    #Retrieves Google search results from the input dictionary, defaulting to an empty list.
    google_diagnoses = results_dict.get("Google Search Results", [])
    #Retrieves PubMed search results from the input dictionary.
    pubmed_diagnoses = results_dict.get("Medical Source (PubMed) Results", [])

    #Initialize lists to hold clean text extracted from URLs.
    #Initializes a list to store cleaned text from Google search results.
    google_texts = []
    #Only attempt to extract text if Google results are present and not just the "No relevant" message.
    #Checks if Google diagnoses are available and not just a "no results" message.
    if google_diagnoses and "No relevant" not in google_diagnoses[0]:
        #Extracts and cleans text from Google result URLs.
        google_texts = extract_clean_text(google_diagnoses, source_type="google")

    #Initializes a list to store cleaned text from PubMed search results.
    pubmed_texts = []
    #Only attempt to extract text if PubMed results are present and not just the "No relevant" message.
    #Checks if PubMed diagnoses are available and not just a "no results" message.
    if pubmed_diagnoses and "No relevant" not in pubmed_diagnoses[0]:
        #Extracts and cleans text from PubMed result URLs.
        pubmed_texts = extract_clean_text(pubmed_diagnoses, source_type="pubmed")

    #Generate Google Keywords plot.
    #Only if sufficient text was extracted from Google sources.
    #Checks if there is sufficient cleaned text from Google results.
    if google_texts:
        #Performs LDA topic modeling on Google texts.
        google_topics = lda_topic_modeling(google_texts)
        #Creates a keyword bar plot for Google topics and stores the figure.
        figures["Google Keywords"] = create_keywords_bar_plot(google_topics, "Google")
    #If no sufficient Google text is available.
    else:
        figures["Google Keywords"] = None #Sets the Google Keywords figure to None.

    #Generate PubMed Keywords plot.
    #Only if sufficient text was extracted from PubMed sources.
    #Checks if there is sufficient cleaned text from PubMed results.
    if pubmed_texts:
        #Performs LDA topic modeling on PubMed texts.
        pubmed_topics = lda_topic_modeling(pubmed_texts)
        #Creates a keyword bar plot for PubMed topics and stores the figure.
        figures["PubMed Keywords"] = create_keywords_bar_plot(pubmed_topics, "PubMed")
    #If no sufficient PubMed text is available.
    else:
        figures["PubMed Keywords"] = None #Sets the PubMed Keywords figure to None.

    #Prepare dataframes for reliability assessment.
    #Estimates reliability for Google search results and creates a DataFrame.
    google_df = estimate_reliability(google_diagnoses, "google")
    #Estimates reliability for PubMed search results and creates a DataFrame.
    pubmed_df = estimate_reliability(pubmed_diagnoses, "pubmed")

    #Generate Reliability Heatmap.
    #Only if there's any data in either Google or PubMed dataframes for reliability.
    #Checks if either the Google or PubMed reliability DataFrames are not empty.
    if not google_df.empty or not pubmed_df.empty:
        #Creates a reliability heatmap and stores the figure.
        figures["Reliability Heatmap"] = create_reliability_heatmap(google_df, pubmed_df)
    #If both reliability DataFrames are empty.
    else:
        figures["Reliability Heatmap"] = None #Sets the Reliability Heatmap figure to None.

    #Returns the dictionary containing all generated figures (or None for those not generated).
    return figures


#This block is for local testing of MismatchMD1.py itself.
#It allows you to run this script directly and see results/plots in separate windows.
#Ensures the following code only runs when the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    #Prints a message indicating direct execution for testing.
    print("Running MismatchMD1.py directly for testing:")
    #Prompts the user to enter test symptoms.
    test_symptoms = input("Enter test symptoms (e.g., 'fever, cough'): ")
    #Calls the main function to get search results based on user-provided symptoms.
    results = get_symptom_results(test_symptoms)

    #Prints a header for the combined text results.
    print("\n--- Combined Text Results ---")
    #Iterates through each source (e.g., "Google Search Results", "Medical Source (PubMed) Results") and its list of information.
    for source, info_list in results.items():
        #Prints the name of the current source.
        print(f"\n** {source}: **")
        #Iterates through each item in the info list, with an index starting from 1.
        for i, item in enumerate(info_list, 1):
            #Prints the numbered result and its content.
            print(f"Result {i}: \n{item}\n")

    #Prints a header for the visualization generation section.
    print("\n--- Generating Visualizations (may open windows) ---")
    #Get the dictionary of figures from the visualization function.
    #Calls the visualization function to generate plots and retrieve them as a dictionary.
    figures = visualize_symptom_analysis(results)

    #Loop through the figures and display them in separate Matplotlib windows.
    #Iterates through the dictionary of generated figures (title and figure object).
    for title, fig in figures.items():
        #Checks if a figure object exists (i.e., not None).
        if fig:
            #Prints a message indicating which plot is being displayed.
            print(f"Displaying {title}...")
            #Set the current figure to be displayed.
            #Activates the specific Matplotlib figure to be displayed.
            plt.figure(fig.number)
            #Show the plot in a new window.
            #Displays the plot in a new window.
            plt.show()
            #Close the figure to free up memory after displaying.
            #Closes the displayed figure to release memory.
            plt.close(fig)
        #If the figure object is None.
        else:
            #Prints a message indicating that no data was available for the plot.
            print(f"No data to display for {title}.")
