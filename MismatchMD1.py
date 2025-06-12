# MismatchMD1.py

# %% (You can keep this if you want, it's a Jupyter/VS Code cell marker)
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time

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

    @staticmethod # Add this decorator
    def get_diagnosis_from_google(symptoms):
        self_google_diagnoses = []
        try:
            search_query = f"self {symptoms} diagnosis"

            for i, url in enumerate(search(search_query, num=10, stop=10, pause=2)): # Reduced num/stop for quicker demo
                # print(f"  Fetching: {url}") # For debugging
                try:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, "html.parser")

                    source_title = soup.title.string if soup.title else "Untitled"

                    for p in soup.find_all("p"):
                        text = p.get_text().strip()
                        if "diagnosis" in text.lower() or "condition" in text.lower():
                            formatted_text = f"[From: {source_title}]\n{text[:500]}...\n[URL: {url}]" # Added ... for truncation
                            self_google_diagnoses.append(formatted_text)
                            # Only take one relevant paragraph per URL to avoid too much text
                            break

                except requests.exceptions.RequestException as req_e:
                    # print(f"  Network/HTTP Error processing {url}: {req_e}")
                    continue
                except Exception as e:
                    # print(f"  Error processing {url}: {str(e)}")
                    continue
                time.sleep(1) # Small delay to be polite to Google

        except Exception as e:
            # print(f"Search error: {str(e)}")
            pass # Suppress general search errors for cleaner output

        unique_results = MedicalMismatchDetector.deduplicate_results(self_google_diagnoses)
        return unique_results[:5] # Return fewer for brevity in demo

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
                'retmax': 10
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            id_list = data['esearchresult'].get('idlist', [])
            if not id_list:
                # print("No PubMed IDs found for the given symptoms.")
                return [] # Return empty if no IDs

            for article_id in id_list[:5]: # Reduced to top 5 for quicker demo
                try:
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    summary_params = {
                        'db': 'pubmed',
                        'id': article_id,
                        'retmode': 'json'
                    }

                    summary_response = requests.get(summary_url, params=summary_params)
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

        return self_pubmed_diagnoses


# NEW FUNCTION: This is the function app.py will call
def get_symptom_results(user_symptoms):
    all_results = {}

    # Get Google results
    google_diagnoses = MedicalMismatchDetector.get_diagnosis_from_google(user_symptoms)
    if google_diagnoses:
        all_results["Google Search Results"] = google_diagnoses
    else:
        all_results["Google Search Results"] = ["No relevant Google results found."]


    # Get PubMed results
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
