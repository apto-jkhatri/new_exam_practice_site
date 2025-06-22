# DBT Documentation Scraper and Vectorizer

This repository contains a Python application designed to scrape DBT (Data Build Tool) documentation, vectorize its content, and create a searchable index using FAISS. This can be used to build a knowledge base for DBT documentation, enabling efficient search and retrieval of information.

## Features

*   **DBT Documentation Scraping**: Fetches documentation content from specified sources.
*   **Text Vectorization**: Converts scraped text into numerical vector representations using `local_vectorizer.py`.
*   **FAISS Indexing**: Creates an efficient similarity search index using FAISS for fast retrieval of relevant documentation.
*   **Local and Remote Processing**: Supports both local and potentially remote (raw text) processing of DBT documentation.

## Project Structure

*   [`app.py`](app.py): The main application file, likely containing the core logic for interacting with the scraped data and FAISS index.
*   [`scraper_and_vectorizer.py`](scraper_and_vectorizer.py): Script responsible for scraping DBT documentation and performing the vectorization process.
*   [`local_vectorizer.py`](local_vectorizer.py): Contains the implementation for text vectorization.
*   [`dbt_docs_links.txt`](dbt_docs_links.txt): A file likely containing URLs or links to DBT documentation pages to be scraped.
*   [`dbt_docs_local_raw_text.json`](dbt_docs_local_raw_text.json): Stores raw text content of locally processed DBT documentation.
*   [`dbt_docs_raw_text.json`](dbt_docs_raw_text.json): Stores raw text content of DBT documentation (possibly from a remote source).
*   [`faiss_dbt_docs_local_embeddings_index/`](faiss_dbt_docs_local_embeddings_index/): Directory containing the FAISS index files for local DBT documentation embeddings.
    *   [`index.faiss`](faiss_dbt_docs_local_embeddings_index/index.faiss): The FAISS index file.
    *   [`index.pkl`](faiss_dbt_docs_local_embeddings_index/index.pkl): A Python pickle file, likely storing metadata or the FAISS index itself.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    You will need to install the required Python packages. A `requirements.txt` file is typically used for this. If one is not present, you will need to identify and install the necessary libraries (e.g., `beautifulsoup4`, `faiss-cpu`, `transformers`, `scikit-learn`, `numpy`).

    *Self-correction: I should mention that a `requirements.txt` file is usually needed and the user might need to create one.*

    ```bash
    # If you have a requirements.txt file:
    pip install -r requirements.txt

    # Otherwise, you might need to install these manually (example):
    pip install beautifulsoup4 requests faiss-cpu transformers numpy scikit-learn
    ```

## Usage

1.  **Prepare DBT Documentation Links**:
    Ensure `dbt_docs_links.txt` contains the URLs of the DBT documentation pages you wish to scrape, one URL per line.

2.  **Scrape and Vectorize Documentation**:
    Run the `scraper_and_vectorizer.py` script to process the documentation and create the FAISS index.

    ```bash
    python scraper_and_vectorizer.py
    ```

3.  **Run the Application**:
    Execute the main application file (`app.py`). Depending on its functionality, this might start a web server or perform a specific task.

    ```bash
    python app.py
    ```

    *Further instructions on how to interact with `app.py` would depend on its specific implementation (e.g., if it's a Flask app, how to access its endpoints).*

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.