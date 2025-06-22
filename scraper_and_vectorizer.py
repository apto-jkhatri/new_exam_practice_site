import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import json # For potentially saving raw scraped data

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document # To create Langchain Document objects
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

BASE_URL = "https://docs.getdbt.com/"
# We'll start from the references overview, but you might want to adjust
START_URL = "https://docs.getdbt.com/reference/references-overview"
# Or more broadly, from a higher level page if you want more content
# START_URL = "https://docs.getdbt.com/docs/introduction"

ALLOWED_DOMAIN = "docs.getdbt.com"
VECTOR_STORE_PATH = "faiss_dbt_docs_index"
RAW_TEXT_CACHE_PATH = "dbt_docs_raw_text.json" # To cache raw scraping results

# --- Helper Functions ---

def is_valid_url(url, base_domain):
    """Checks if the URL is valid, within the domain, and not an anchor/mailto etc."""
    parsed_url = urlparse(url)
    return (parsed_url.scheme in ["http", "https"] and
            parsed_url.netloc == base_domain and
            not parsed_url.fragment and  # No anchors like #section
            not parsed_url.path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.zip', '.pdf')) and # No media
            "mailto:" not in url and
            "javascript:" not in url)

def scrape_page_content(url):
    """Fetches and extracts meaningful text content from a page."""
    print(f"Scraping: {url}")
    try:
        headers = {'User-Agent': 'dbt-exam-prep-scraper/1.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()  # Raises an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to find the main content area. This selector might need adjustment
        # based on dbt docs structure. Inspect the page to find the best one.
        # Common selectors: 'main', 'article', 'div.content', 'div.main-content'
        # For dbt docs, 'article.theme-doc-markdown.markdown' seems to hold the main text.
        content_area = soup.find("article", class_="theme-doc-markdown markdown")

        if not content_area:
            # Fallback if the specific class isn't found, try a more generic 'main'
            content_area = soup.find("main")
            if not content_area:
                print(f"Warning: Could not find main content area for {url}. Skipping text extraction.")
                return "" # Return empty string if no suitable content area

        # Remove nav, footer, scripts, styles if they are inside content_area
        for unwelcome_tag in content_area.find_all(['nav', 'footer', 'script', 'style', 'aside', 'button', '.tocCollapsible_ETCw']):
            unwelcome_tag.decompose()

        # Extract text from relevant tags (p, h1-h6, li, code, pre)
        texts = []
        for tag in content_area.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'code', 'pre']):
            # Get text and clean it up
            text = tag.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
        
        page_text = "\n".join(texts)
        # print(f"Extracted {len(page_text)} characters from {url}")
        return page_text

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return ""

def scrape_website(start_url, base_url, allowed_domain, max_pages=100):
    """
    Crawls a website starting from start_url, staying within allowed_domain,
    and extracts text content.
    """
    queue = deque([start_url])
    visited_urls = {start_url}
    all_page_data = [] # List of dictionaries: {'url': url, 'text': text}
    pages_scraped = 0

    while queue and pages_scraped < max_pages:
        current_url = queue.popleft()
        
        text_content = scrape_page_content(current_url)
        if text_content:
            all_page_data.append({"url": current_url, "text": text_content})
            pages_scraped += 1

        # Find new links on the current page
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for links primarily within the main content or navigation areas
            # This helps avoid less relevant links like "edit this page" or external blogs
            link_areas = soup.find_all(['article', 'nav', 'aside']) # Common areas with navigation links
            if not link_areas: # If no specific areas, search the whole body
                link_areas = [soup.body]

            for area in link_areas:
                if area: # Ensure area is not None
                    for a_tag in area.find_all("a", href=True):
                        href = a_tag['href']
                        full_url = urljoin(current_url, href) # Handles relative links

                        if is_valid_url(full_url, allowed_domain) and full_url not in visited_urls:
                            if len(visited_urls) < max_pages * 2: # Limit total URLs to explore
                                visited_urls.add(full_url)
                                queue.append(full_url)
        except requests.RequestException as e:
            print(f"Error fetching links from {current_url}: {e}")
        except Exception as e:
            print(f"Error processing links on {current_url}: {e}")
            
        time.sleep(0.1) # Be respectful to the server

    return all_page_data

# --- Main Execution ---
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with OPENAI_API_KEY='your_key_here'")
        exit()

    all_scraped_data = []

    # Check if we have cached raw text
    if os.path.exists(RAW_TEXT_CACHE_PATH):
        print(f"Loading raw scraped data from {RAW_TEXT_CACHE_PATH}...")
        with open(RAW_TEXT_CACHE_PATH, 'r', encoding='utf-8') as f:
            all_scraped_data = json.load(f)
        print(f"Loaded {len(all_scraped_data)} pages from cache.")
    else:
        print("Starting website scraping process...")
        # Adjust max_pages as needed. dbt docs are extensive!
        # For a full scrape, this could be 1000s, but start small for testing.
        all_scraped_data = scrape_website(START_URL, BASE_URL, ALLOWED_DOMAIN, max_pages=50) # Increased max_pages
        print(f"Scraped {len(all_scraped_data)} pages in total.")
        
        # Cache the raw scraped data
        with open(RAW_TEXT_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_scraped_data, f, indent=2)
        print(f"Raw scraped data cached to {RAW_TEXT_CACHE_PATH}")

    if not all_scraped_data:
        print("No data scraped. Exiting.")
        exit()

    # Convert scraped data into Langchain Document objects
    documents = []
    for page_data in all_scraped_data:
        # We add the URL as metadata to each document (and later, to each chunk)
        # This is useful for citing sources or understanding where the info came from.
        doc = Document(page_content=page_data['text'], metadata={"source": page_data['url']})
        documents.append(doc)
    
    print(f"Created {len(documents)} Langchain Document objects.")

    # 2. Split documents into chunks
    # This splitter tries to keep paragraphs/sentences together.
    # chunk_size is in characters (not tokens, though it's related).
    # chunk_overlap helps maintain context between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Max size of a chunk
        chunk_overlap=200, # Overlap between chunks
        length_function=len
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunked_documents)} chunks.")

    if not chunked_documents:
        print("No chunks created. This might happen if scraped content was too short or empty.")
        exit()

    # 3. Create embeddings and store in FAISS
    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings() # By default, uses 'text-embedding-ada-002' or newer

    print(f"Creating FAISS vector store from {len(chunked_documents)} chunks...")
    # This can take some time and will make API calls to OpenAI for each chunk
    try:
        vector_store = FAISS.from_documents(chunked_documents, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store created and saved to {VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"Error creating or saving FAISS vector store: {e}")
        print("This could be due to API key issues, rate limits, or empty documents.")
        exit()

    # Optional: Test the vector store
    print("\n--- Testing Vector Store ---")
    try:
        # Load it back (as you would in your Streamlit app)
        loaded_vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True) # allow_dangerous_deserialization for FAISS with pickle
        
        query = "What is a dbt model?"
        results = loaded_vector_store.similarity_search(query, k=2) # Get top 2 similar chunks
        
        if results:
            print(f"\nTop results for query: '{query}'")
            for i, doc in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Source: {doc.metadata.get('source', 'N/A')}")
                print(f"Content snippet: {doc.page_content[:300]}...") # Print first 300 chars
        else:
            print(f"No results found for query: '{query}'")
            
    except Exception as e:
        print(f"Error testing FAISS vector store: {e}")

    print("\nProcess complete!")