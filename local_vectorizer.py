import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import json
import re # For cleaning up text

# Langchain components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # For local embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv # Still good for Groq key later, but not used in this script

# --- Configuration ---
# load_dotenv() # We'll activate this when we integrate Groq

BASE_URL = "https://docs.getdbt.com/"
# Let's try a broader start, but be mindful of scraping scope
START_URLS = [
    'https://docs.getdbt.com/docs/introduction',
    'https://docs.getdbt.com/guides/overview',
    'https://docs.getdbt.com/reference/references-overview',
    'https://docs.getdbt.com/reference/dbt_project.yml',
    'https://docs.getdbt.com/reference/model-configs',
    'https://docs.getdbt.com/docs/build/models',
    'https://docs.getdbt.com/docs/build/sources',
    'https://docs.getdbt.com/docs/build/seeds',
    'https://docs.getdbt.com/docs/build/snapshots',
    'https://docs.getdbt.com/docs/build/tests',
    'https://docs.getdbt.com/docs/build/macros',
    'https://docs.getdbt.com/reference/global-configs/profiles.yml',
    'https://docs.getdbt.com/reference/commands/run',
    'https://docs.getdbt.com/reference/advanced-config-usage',
    'https://docs.getdbt.com/reference/analysis-properties',
    'https://docs.getdbt.com/reference/artifacts/catalog-json',
    'https://docs.getdbt.com/reference/artifacts/dbt-artifacts',
    'https://docs.getdbt.com/reference/artifacts/manifest-json',
    'https://docs.getdbt.com/reference/artifacts/other-artifacts',
    'https://docs.getdbt.com/reference/artifacts/run-results-json',
    'https://docs.getdbt.com/reference/artifacts/sl-manifest',
    'https://docs.getdbt.com/reference/artifacts/sources-json',
    'https://docs.getdbt.com/reference/commands/build',
    'https://docs.getdbt.com/reference/commands/clean',
    'https://docs.getdbt.com/reference/commands/clone',
    'https://docs.getdbt.com/reference/commands/cmd-docs',
    'https://docs.getdbt.com/reference/commands/compile',
    'https://docs.getdbt.com/reference/commands/dbt-environment',
    'https://docs.getdbt.com/reference/commands/debug',
    'https://docs.getdbt.com/reference/commands/deps',
    'https://docs.getdbt.com/reference/commands/init',
    'https://docs.getdbt.com/reference/commands/invocation',
    'https://docs.getdbt.com/reference/commands/list',
    'https://docs.getdbt.com/reference/commands/parse',
    'https://docs.getdbt.com/reference/commands/retry',
    'https://docs.getdbt.com/reference/commands/rpc',
    'https://docs.getdbt.com/reference/commands/run',
    'https://docs.getdbt.com/reference/commands/run-operation',
    'https://docs.getdbt.com/reference/commands/seed',
    'https://docs.getdbt.com/reference/commands/show',
    'https://docs.getdbt.com/reference/commands/snapshot',
    'https://docs.getdbt.com/reference/commands/source',
    'https://docs.getdbt.com/reference/commands/test',
    'https://docs.getdbt.com/reference/commands/version',
    'https://docs.getdbt.com/reference/configs-and-properties',
    'https://docs.getdbt.com/reference/data-test-configs',
    'https://docs.getdbt.com/reference/database-permissions/about-database-permissions',
    'https://docs.getdbt.com/reference/database-permissions/databricks-permissions',
    'https://docs.getdbt.com/reference/database-permissions/postgres-permissions',
    'https://docs.getdbt.com/reference/database-permissions/redshift-permissions',
    'https://docs.getdbt.com/reference/database-permissions/snowflake-permissions',
    'https://docs.getdbt.com/reference/dbt-classes',
    'https://docs.getdbt.com/reference/dbt-commands',
    'https://docs.getdbt.com/reference/dbt-jinja-functions',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/adapter',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/as_bool',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/as_native',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/as_number',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/builtins',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/config',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/cross-database-macros',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/dbt-project-yml-context',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/dbt-properties-yml-context',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/dbt_version',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/debug-method',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/dispatch',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/doc',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/env_var',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/exceptions',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/execute',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/flags',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/fromjson',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/fromyaml',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/graph',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/invocation_id',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/local_md5',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/log',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/model',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/modules',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/on-run-end-context',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/print',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/profiles-yml-context',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/project_name',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/ref',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/return',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/run_query',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/run_started_at',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/schema',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/schemas',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/selected_resources',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/set',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/source',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/statement-blocks',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/target',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/this',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/thread_id',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/tojson',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/toyaml',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/var',
    'https://docs.getdbt.com/reference/dbt-jinja-functions/zip',
    'https://docs.getdbt.com/reference/dbt_project.yml',
    'https://docs.getdbt.com/reference/dbtignore',
    'https://docs.getdbt.com/reference/define-configs',
    'https://docs.getdbt.com/reference/define-properties',
    'https://docs.getdbt.com/reference/deprecations',
    'https://docs.getdbt.com/reference/events-logging',
    'https://docs.getdbt.com/reference/exit-codes',
    'https://docs.getdbt.com/reference/exposure-properties',
    'https://docs.getdbt.com/reference/global-configs/about-global-configs',
    'https://docs.getdbt.com/reference/global-configs/adapter-behavior-changes',
    'https://docs.getdbt.com/reference/global-configs/behavior-changes',
    'https://docs.getdbt.com/reference/global-configs/cache',
    'https://docs.getdbt.com/reference/global-configs/command-line-options',
    'https://docs.getdbt.com/reference/global-configs/databricks-changes',
    'https://docs.getdbt.com/reference/global-configs/environment-variable-configs',
    'https://docs.getdbt.com/reference/global-configs/failing-fast',
    'https://docs.getdbt.com/reference/global-configs/indirect-selection',
    'https://docs.getdbt.com/reference/global-configs/json-artifacts',
    'https://docs.getdbt.com/reference/global-configs/logs',
    'https://docs.getdbt.com/reference/global-configs/parsing',
    'https://docs.getdbt.com/reference/global-configs/print-output',
    'https://docs.getdbt.com/reference/global-configs/project-flags',
    'https://docs.getdbt.com/reference/global-configs/record-timing-info',
    'https://docs.getdbt.com/reference/global-configs/redshift-changes',
    'https://docs.getdbt.com/reference/global-configs/resource-type',
    'https://docs.getdbt.com/reference/global-configs/snowflake-changes',
    'https://docs.getdbt.com/reference/global-configs/usage-stats',
    'https://docs.getdbt.com/reference/global-configs/version-compatibility',
    'https://docs.getdbt.com/reference/global-configs/warnings',
    'https://docs.getdbt.com/reference/macro-properties',
    'https://docs.getdbt.com/reference/model-configs',
    'https://docs.getdbt.com/reference/model-properties',
    'https://docs.getdbt.com/reference/node-selection/configure-state',
    'https://docs.getdbt.com/reference/node-selection/defer',
    'https://docs.getdbt.com/reference/node-selection/exclude',
    'https://docs.getdbt.com/reference/node-selection/graph-operators',
    'https://docs.getdbt.com/reference/node-selection/methods',
    'https://docs.getdbt.com/reference/node-selection/putting-it-together',
    'https://docs.getdbt.com/reference/node-selection/set-operators',
    'https://docs.getdbt.com/reference/node-selection/state-comparison-caveats',
    'https://docs.getdbt.com/reference/node-selection/state-selection',
    'https://docs.getdbt.com/reference/node-selection/syntax',
    'https://docs.getdbt.com/reference/node-selection/test-selection-examples',
    'https://docs.getdbt.com/reference/node-selection/yaml-selectors',
    'https://docs.getdbt.com/reference/parsing',
    'https://docs.getdbt.com/reference/programmatic-invocations',
    'https://docs.getdbt.com/reference/project-configs/analysis-paths',
    'https://docs.getdbt.com/reference/project-configs/asset-paths',
    'https://docs.getdbt.com/reference/project-configs/clean-targets',
    'https://docs.getdbt.com/reference/project-configs/config-version',
    'https://docs.getdbt.com/reference/project-configs/dispatch-config',
    'https://docs.getdbt.com/reference/project-configs/docs-paths',
    'https://docs.getdbt.com/reference/project-configs/macro-paths',
    'https://docs.getdbt.com/reference/project-configs/model-paths',
    'https://docs.getdbt.com/reference/project-configs/name',
    'https://docs.getdbt.com/reference/project-configs/on-run-start-on-run-end',
    'https://docs.getdbt.com/reference/project-configs/packages-install-path',
    'https://docs.getdbt.com/reference/project-configs/profile',
    'https://docs.getdbt.com/reference/project-configs/query-comment',
    'https://docs.getdbt.com/reference/project-configs/quoting',
    'https://docs.getdbt.com/reference/project-configs/require-dbt-version',
    'https://docs.getdbt.com/reference/project-configs/seed-paths',
    'https://docs.getdbt.com/reference/project-configs/snapshot-paths',
    'https://docs.getdbt.com/reference/project-configs/test-paths',
    'https://docs.getdbt.com/reference/project-configs/version',
    'https://docs.getdbt.com/reference/references-overview',
    'https://docs.getdbt.com/reference/resource-configs/access',
    'https://docs.getdbt.com/reference/resource-configs/alias',
    'https://docs.getdbt.com/reference/resource-configs/athena-configs',
    'https://docs.getdbt.com/reference/resource-configs/azuresynapse-configs',
    'https://docs.getdbt.com/reference/resource-configs/batch-size',
    'https://docs.getdbt.com/reference/resource-configs/begin',
    'https://docs.getdbt.com/reference/resource-configs/bigquery-configs',
    'https://docs.getdbt.com/reference/resource-configs/check_cols',
    'https://docs.getdbt.com/reference/resource-configs/clickhouse-configs',
    'https://docs.getdbt.com/reference/resource-configs/column_types',
    'https://docs.getdbt.com/reference/resource-configs/contract',
    'https://docs.getdbt.com/reference/resource-configs/database',
    'https://docs.getdbt.com/reference/resource-configs/databricks-configs',
    'https://docs.getdbt.com/reference/resource-configs/dbt_valid_to_current',
    'https://docs.getdbt.com/reference/resource-configs/delimiter',
    'https://docs.getdbt.com/reference/resource-configs/docs',
    'https://docs.getdbt.com/reference/resource-configs/doris-configs',
    'https://docs.getdbt.com/reference/resource-configs/duckdb-configs',
    'https://docs.getdbt.com/reference/resource-configs/enabled',
    'https://docs.getdbt.com/reference/resource-configs/event-time',
    'https://docs.getdbt.com/reference/resource-configs/fabric-configs',
    'https://docs.getdbt.com/reference/resource-configs/fabricspark-configs',
    'https://docs.getdbt.com/reference/resource-configs/fail_calc',
    'https://docs.getdbt.com/reference/resource-configs/firebolt-configs',
    'https://docs.getdbt.com/reference/resource-configs/freshness',
    'https://docs.getdbt.com/reference/resource-configs/full_refresh',
    'https://docs.getdbt.com/reference/resource-configs/grants',
    'https://docs.getdbt.com/reference/resource-configs/greenplum-configs',
    'https://docs.getdbt.com/reference/resource-configs/group',
    'https://docs.getdbt.com/reference/resource-configs/hard-deletes',
    'https://docs.getdbt.com/reference/resource-configs/ibm-netezza-config',
    'https://docs.getdbt.com/reference/resource-configs/impala-configs',
    'https://docs.getdbt.com/reference/resource-configs/infer-configs',
    'https://docs.getdbt.com/reference/resource-configs/invalidate_hard_deletes',
    'https://docs.getdbt.com/reference/resource-configs/limit',
    'https://docs.getdbt.com/reference/resource-configs/lookback',
    'https://docs.getdbt.com/reference/resource-configs/materialize-configs',
    'https://docs.getdbt.com/reference/resource-configs/materialized',
    'https://docs.getdbt.com/reference/resource-configs/meta',
    'https://docs.getdbt.com/reference/resource-configs/mindsdb-configs',
    'https://docs.getdbt.com/reference/resource-configs/mssql-configs',
    'https://docs.getdbt.com/reference/resource-configs/on_configuration_change',
    'https://docs.getdbt.com/reference/resource-configs/oracle-configs',
    'https://docs.getdbt.com/reference/resource-configs/persist_docs',
    'https://docs.getdbt.com/reference/resource-configs/plus-prefix',
    'https://docs.getdbt.com/reference/resource-configs/postgres-configs',
    'https://docs.getdbt.com/reference/resource-configs/pre-hook-post-hook',
    'https://docs.getdbt.com/reference/resource-configs/quote_columns',
    'https://docs.getdbt.com/reference/resource-configs/redshift-configs',
    'https://docs.getdbt.com/reference/resource-configs/resource-configs',
    'https://docs.getdbt.com/reference/resource-configs/resource-path',
    'https://docs.getdbt.com/reference/resource-configs/schema',
    'https://docs.getdbt.com/reference/resource-configs/severity',
    'https://docs.getdbt.com/reference/resource-configs/singlestore-configs',
    'https://docs.getdbt.com/reference/resource-configs/snapshot_meta_column_names',
    'https://docs.getdbt.com/reference/resource-configs/snapshot_name',
    'https://docs.getdbt.com/reference/resource-configs/snapshots-jinja-legacy',
    'https://docs.getdbt.com/reference/resource-configs/snowflake-configs',
    'https://docs.getdbt.com/reference/resource-configs/spark-configs',
    'https://docs.getdbt.com/reference/resource-configs/sql_header',
    'https://docs.getdbt.com/reference/resource-configs/starrocks-configs',
    'https://docs.getdbt.com/reference/resource-configs/store_failures',
    'https://docs.getdbt.com/reference/resource-configs/store_failures_as',
    'https://docs.getdbt.com/reference/resource-configs/strategy',
    'https://docs.getdbt.com/reference/resource-configs/tags',
    'https://docs.getdbt.com/reference/resource-configs/target_database',
    'https://docs.getdbt.com/reference/resource-configs/target_schema',
    'https://docs.getdbt.com/reference/resource-configs/teradata-configs',
    'https://docs.getdbt.com/reference/resource-configs/trino-configs',
    'https://docs.getdbt.com/reference/resource-configs/unique_key',
    'https://docs.getdbt.com/reference/resource-configs/updated_at',
    'https://docs.getdbt.com/reference/resource-configs/upsolver-configs',
    'https://docs.getdbt.com/reference/resource-configs/vertica-configs',
    'https://docs.getdbt.com/reference/resource-configs/watsonx-presto-config',
    'https://docs.getdbt.com/reference/resource-configs/watsonx-spark-config',
    'https://docs.getdbt.com/reference/resource-configs/where',
    'https://docs.getdbt.com/reference/resource-configs/yellowbrick-configs',
    'https://docs.getdbt.com/reference/resource-properties/arguments',
    'https://docs.getdbt.com/reference/resource-properties/columns',
    'https://docs.getdbt.com/reference/resource-properties/concurrent_batches',
    'https://docs.getdbt.com/reference/resource-properties/config',
    'https://docs.getdbt.com/reference/resource-properties/constraints',
    'https://docs.getdbt.com/reference/resource-properties/data-formats',
    'https://docs.getdbt.com/reference/resource-properties/data-tests',
    'https://docs.getdbt.com/reference/resource-properties/data-types',
    'https://docs.getdbt.com/reference/resource-properties/database',
    'https://docs.getdbt.com/reference/resource-properties/deprecation_date',
    'https://docs.getdbt.com/reference/resource-properties/description',
    'https://docs.getdbt.com/reference/resource-properties/external',
    'https://docs.getdbt.com/reference/resource-properties/freshness',
    'https://docs.getdbt.com/reference/resource-properties/identifier',
    'https://docs.getdbt.com/reference/resource-properties/latest_version',
    'https://docs.getdbt.com/reference/resource-properties/loader',
    'https://docs.getdbt.com/reference/resource-properties/model_name',
    'https://docs.getdbt.com/reference/resource-properties/overrides',
    'https://docs.getdbt.com/reference/resource-properties/quoting',
    'https://docs.getdbt.com/reference/resource-properties/schema',
    'https://docs.getdbt.com/reference/resource-properties/unit-test-input',
    'https://docs.getdbt.com/reference/resource-properties/unit-test-overrides',
    'https://docs.getdbt.com/reference/resource-properties/unit-testing-versions',
    'https://docs.getdbt.com/reference/resource-properties/unit-tests',
    'https://docs.getdbt.com/reference/resource-properties/versions',
    'https://docs.getdbt.com/reference/seed-configs',
    'https://docs.getdbt.com/reference/seed-properties',
    'https://docs.getdbt.com/reference/snapshot-configs',
    'https://docs.getdbt.com/reference/snapshot-properties',
    'https://docs.getdbt.com/reference/source-configs',
    'https://docs.getdbt.com/reference/source-properties'
]
ALLOWED_DOMAIN = "docs.getdbt.com"
VECTOR_STORE_PATH = "faiss_dbt_docs_local_embeddings_index"
RAW_TEXT_CACHE_PATH = "dbt_docs_local_raw_text.json"

# --- Helper Functions ---

def is_valid_url(url, base_domain):
    parsed_url = urlparse(url)
    return (parsed_url.scheme in ["http", "https"] and
            parsed_url.netloc == base_domain and
            not parsed_url.fragment and
            not any(parsed_url.path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.zip', '.pdf', '.ico', '.svg']) and
            "mailto:" not in url and
            "javascript:" not in url and
            # Avoid API reference and some other less relevant sections for general Q&A
            "/api/ff/" not in parsed_url.path and
            "/terms" not in parsed_url.path and
            "/community" not in parsed_url.path and # Community forum links, etc.
            "/blog" not in parsed_url.path and
            "/changelog" not in parsed_url.path
            )

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.replace("Copy code", "").strip() # Remove common "Copy code" button text
    return text

def extract_content_with_structure(soup_content_area):
    """
    Extracts text trying to maintain some structure using headings.
    For "nesting", we'll prepend parent heading text to content under it.
    This is a simplified approach to "nesting" for linear text.
    """
    extracted_elements = []
    current_headings = [""] * 6  # For h1 to h6

    for element in soup_content_area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'table']):
        text = ""
        is_heading = False
        heading_level = 0

        if element.name.startswith('h') and element.name[1].isdigit():
            level = int(element.name[1])
            if 1 <= level <= 6:
                heading_level = level
                is_heading = True
                text = clean_text(element.get_text(separator=" ", strip=True))
                current_headings[level-1] = text
                # Reset lower-level headings
                for i in range(level, 6):
                    current_headings[i] = ""
        elif element.name == 'p' or element.name == 'li':
            text = clean_text(element.get_text(separator=" ", strip=True))
        elif element.name == 'pre': # Code blocks
            code_tag = element.find('code')
            if code_tag:
                text = code_tag.get_text(strip=False) # Preserve formatting within code
                text = f"\n```\n{text.strip()}\n```\n" # Markdown for code
            else:
                text = f"\n```\n{element.get_text(strip=True)}\n```\n"
        elif element.name == 'table':
            # Basic table to text conversion (can be improved)
            rows = []
            for row_el in element.find_all('tr'):
                cols = [clean_text(col.get_text(separator=" ", strip=True)) for col in row_el.find_all(['td', 'th'])]
                rows.append(" | ".join(cols))
            text = "\n" + "\n".join(rows) + "\n"

        if text:
            # Prepend relevant parent headings for context (simplified nesting)
            context_headings = [h for h in current_headings[:heading_level if is_heading else 6] if h]
            prefix = " -> ".join(context_headings[:-1]) if not is_heading and len(context_headings) > 1 else ""
            if prefix and not is_heading : # Add prefix only to non-heading content if prefix exists
                 extracted_elements.append(f"{prefix}\n{text}")
            elif is_heading: # Add heading itself
                 extracted_elements.append(" -> ".join(context_headings))
            else: # Content without significant heading prefix (e.g. top-level paragraph)
                 extracted_elements.append(text)


    return "\n\n".join(extracted_elements) # Join elements with double newline for separation

def scrape_page_content_structured(url):
    print(f"Scraping: {url}")
    try:
        headers = {'User-Agent': 'dbt-exam-prep-scraper-local/1.0'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        content_area = soup.find("article", class_="theme-doc-markdown markdown")
        if not content_area:
            content_area = soup.find("main")
            if not content_area:
                print(f"Warning: Could not find main content area for {url}. Skipping text extraction.")
                return ""

        for unwelcome_tag_selector in [
            'nav', 'footer', 'script', 'style', 'aside', 'button',
            '.tocCollapsible_ETCw', 'div.theme-doc-toc-mobile',
            'div.breadcrumbs', 'div.theme-edit-this-page', 'div.pagination-nav',
            'div.margin-vert--lg', # Often contains "Next article" links etc.
            'a.hash-link' # The '#' link icons next to headers
        ]:
            for tag in content_area.select(unwelcome_tag_selector):
                tag.decompose()
        
        page_text = extract_content_with_structure(content_area)
        return page_text.strip()

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return ""

def scrape_website_recursive(start_urls, base_url, allowed_domain, max_pages=200): # Increased max_pages
    queue = deque(start_urls)
    visited_urls = set(start_urls)
    all_page_data = []
    pages_scraped_count = 0

    processed_urls_in_session = set() # To avoid adding URLs multiple times in one run if linked from various places

    while queue and pages_scraped_count < max_pages:
        current_url = queue.popleft()
        
        # Normalize URL to avoid re-scraping due to trailing slash etc.
        parsed_current_url = urlparse(current_url)
        normalized_url = parsed_current_url._replace(query="", fragment="").geturl()
        if normalized_url.endswith('/'):
            normalized_url = normalized_url[:-1]

        if normalized_url in processed_urls_in_session:
            continue
        processed_urls_in_session.add(normalized_url)

        text_content = scrape_page_content_structured(normalized_url)
        if text_content and len(text_content) > 100: # Ensure some meaningful content
            all_page_data.append({"url": normalized_url, "text": text_content})
            pages_scraped_count += 1
            if pages_scraped_count % 10 == 0:
                 print(f"Successfully scraped {pages_scraped_count}/{max_pages} pages...")
        else:
            print(f"Skipping {normalized_url} due to insufficient content.")


        # Find new links only if we haven't hit max_pages
        if pages_scraped_count < max_pages:
            try:
                # Re-fetch for links if needed, or parse from already fetched content
                # For simplicity, let's assume scrape_page_content_structured doesn't modify soup too much for link finding
                headers = {'User-Agent': 'dbt-exam-prep-scraper-local/1.0'}
                response = requests.get(current_url, timeout=10, headers=headers) # current_url not normalized_url here
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                # Focus link finding on main content and navigation sidebars
                link_container_selectors = ["article.theme-doc-markdown.markdown", "main", "nav.theme-doc-sidebar-menu", "aside.theme-doc-sidebar-container"]
                links_found_on_page = set()

                for selector in link_container_selectors:
                    container = soup.select_one(selector)
                    if container:
                        for a_tag in container.find_all("a", href=True):
                            href = a_tag['href']
                            full_url = urljoin(current_url, href)
                            parsed_full_url = urlparse(full_url)
                            normalized_full_url = parsed_full_url._replace(query="", fragment="").geturl()
                            if normalized_full_url.endswith('/'):
                                 normalized_full_url = normalized_full_url[:-1]


                            if is_valid_url(normalized_full_url, allowed_domain) and \
                               normalized_full_url not in visited_urls and \
                               len(visited_urls) < max_pages * 1.5: # Explore a bit more than strictly needed
                                links_found_on_page.add(normalized_full_url)
                
                for link in links_found_on_page:
                    if link not in visited_urls:
                        visited_urls.add(link)
                        queue.append(link)
                        # print(f"  Added to queue: {link}")


            except requests.RequestException as e:
                print(f"Error fetching links from {current_url}: {e}")
            except Exception as e:
                print(f"Error processing links on {current_url}: {e}")
        
        time.sleep(0.2) # Be respectful

    return all_page_data

# --- Main Execution ---
if __name__ == "__main__":
    all_scraped_data = []

    if os.path.exists(RAW_TEXT_CACHE_PATH):
        print(f"Loading raw scraped data from {RAW_TEXT_CACHE_PATH}...")
        try:
            with open(RAW_TEXT_CACHE_PATH, 'r', encoding='utf-8') as f:
                all_scraped_data = json.load(f)
            print(f"Loaded {len(all_scraped_data)} pages from cache.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {RAW_TEXT_CACHE_PATH}. Re-scraping.")
            all_scraped_data = [] # Reset if cache is corrupt

    if not all_scraped_data: # If cache didn't exist, was empty, or corrupt
        print("Starting website scraping process...")
        # Start with a smaller max_pages for testing, e.g., 50-100
        # For full docs, dbt has hundreds of pages.
        all_scraped_data = scrape_website_recursive(START_URLS, BASE_URL, ALLOWED_DOMAIN, max_pages=200) # Adjust as needed
        print(f"Scraped {len(all_scraped_data)} new pages in total.")
        
        if all_scraped_data:
            try:
                with open(RAW_TEXT_CACHE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(all_scraped_data, f, indent=2)
                print(f"Raw scraped data cached to {RAW_TEXT_CACHE_PATH}")
            except Exception as e:
                print(f"Error caching raw data: {e}")
        else:
            print("No new data was scraped.")


    if not all_scraped_data:
        print("No data available (from cache or scraping). Exiting.")
        exit()

    documents = []
    for page_data in all_scraped_data:
        if page_data.get('text') and page_data.get('url'): # Ensure data is valid
            doc = Document(page_content=page_data['text'], metadata={"source": page_data['url']})
            documents.append(doc)
    
    print(f"Created {len(documents)} Langchain Document objects from scraped data.")

    if not documents:
        print("No documents to process for vector store. Exiting.")
        exit()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Aim for chunks that are not too long for LLM context later
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""], # Define preferred split points
        keep_separator=False
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunked_documents)} chunks.")

    if not chunked_documents:
        print("No chunks created. This might happen if scraped content was too short or empty after filtering. Exiting.")
        exit()

    # Use local HuggingFace embeddings (no API key needed for this step)
    print("Initializing local HuggingFace embeddings (sentence-transformers)...")
    # 'all-MiniLM-L6-v2' is small and fast.
    # 'all-mpnet-base-v2' is larger and often better quality.
    # For dbt technical docs, a model good at technical language might be even better if you find one.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have a GPU and PyTorch with CUDA installed
    encode_kwargs = {'normalize_embeddings': False} # Typically False for FAISS with dot product
    
    print(f"Using local embedding model: {model_name}. This may download the model if not cached.")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print(f"Creating FAISS vector store from {len(chunked_documents)} chunks...")
    # This will run locally and can take time depending on CPU and number of chunks.
    # It does NOT use an API key.
    try:
        vector_store = FAISS.from_documents(chunked_documents, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store created and saved to {VECTOR_STORE_PATH}")
    except Exception as e:
        print(f"Error creating or saving FAISS vector store: {e}")
        exit()

    print("\n--- Testing Local Vector Store ---")
    try:
        # Load it back (as you would in your Streamlit app)
        # Ensure you use the same embedding function to load
        loaded_vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True # Necessary for FAISS with pickle
        )
        
        query = "How do I configure a dbt model?"
        results = loaded_vector_store.similarity_search(query, k=3)
        
        if results:
            print(f"\nTop results for query: '{query}'")
            for i, doc_result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Source: {doc_result.metadata.get('source', 'N/A')}")
                print(f"Content snippet: {doc_result.page_content[:350]}...")
        else:
            print(f"No results found for query: '{query}'")
            
    except Exception as e:
        print(f"Error testing FAISS vector store: {e}")

    print("\nLocal scraping and vector database creation complete!")
    print(f"Next step: Integrate this with Streamlit and use Groq API for question generation using the content from '{VECTOR_STORE_PATH}'.")