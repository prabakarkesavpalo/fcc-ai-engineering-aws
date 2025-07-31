# %% [markdown]
# <h1 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4); 
#            color: white; 
#            padding: 20px; 
#            border-radius: 10px; 
#            text-align: center; 
#            font-family: Arial, sans-serif; 
#            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#   Multimodal RAG with Google Gemini Vision API and PostgreSQL
# </h1>

# %% [markdown]
# This notebook demonstrates how to implement a **local multi-modal Retrieval-Augmented Generation (RAG) system** using **Google Gemini Vision API and PostgreSQL with pgvector**. Many documents contain a mixture of content types, including text and images. Traditional RAG applications often lose valuable information captured in images. With the emergence of Multimodal Large Language Models (MLLMs), we can now leverage both text and image data in our RAG systems.
# 
# In this notebook, we'll explore a **local approach** to multi-modal RAG:
# 
# 1. Use **Google Gemini Vision API** to generate multimodal embeddings for both images and text
# 2. Store embeddings in **PostgreSQL with pgvector extension** for efficient similarity search
# 3. Retrieve relevant information using vector similarity search
# 4. Pass raw images and text chunks to **Google Gemini Pro Vision** for answer synthesis
# 
# We'll use the following tools and technologies:
# 
# - **[Google Gemini Vision API](https://ai.google.dev/docs/vision_overview)** for multimodal embeddings and answer generation
# - **[PostgreSQL](https://www.postgresql.org/)** with **[pgvector](https://github.com/pgvector/pgvector)** extension for vector storage and similarity search
# - **[psycopg2](https://pypi.org/project/psycopg2/)** for PostgreSQL database connectivity
# - **[pymupdf](https://pymupdf.readthedocs.io/en/latest/)** to parse images, text, and tables from documents (PDFs)
# - **[google-generativeai](https://pypi.org/project/google-generativeai/)** for interacting with Google's Gemini models
# 
# This approach allows us to create a **completely local** and **cost-effective** RAG system that can understand and utilize both textual and visual information from our documents, while maintaining data privacy and control.
# 
# ## Prerequisites
# 
# Before running this notebook, ensure you have the following installed and configured:
# 
# ### Software Requirements:
# - Python 3.10 or later
# - PostgreSQL (with pgvector extension)
# - Google API key for Gemini
# 
# ### Python Packages:
# - google-generativeai
# - psycopg2-binary
# - pgvector
# - pymupdf
# - numpy
# - tqdm
# - requests
# 
# ### Setup Steps:
# 1. **Install PostgreSQL and pgvector**: Follow instructions for your OS
# 2. **Get Google API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to get your API key
# 3. **Configure environment**: Set your `GOOGLE_API_KEY` environment variable
# 
# Let's get started with building our local multi-modal RAG system!

# %% [markdown]
# ![Multimodal RAG with Amazon Bedrock](imgs/multimodal-rag1.png)

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#    Importing the libs
# </h2>

# %% [markdown]
# 

# %% [markdown]
# Make sure you have installed all the required libraries and services to run this notebook:
# 
# ### Install Python packages:
# ```bash
# cd 03-multimodal-rag
# pip install google-generativeai psycopg2-binary pgvector numpy tqdm requests pymupdf tabula-py
# ```
# 
# ### Install PostgreSQL with pgvector (macOS):
# ```bash
# # Install PostgreSQL
# brew install postgresql
# 
# # Start PostgreSQL service
# brew services start postgresql
# 
# # Install pgvector extension
# git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
# cd pgvector
# make
# make install
# ```
# 
# ### 🔑 Set up Google API Key (IMPORTANT):
# 
# **Option 1: Set environment variable (recommended)**
# ```bash
# # Get your API key from https://makersuite.google.com/app/apikey
# export GOOGLE_API_KEY="your_actual_api_key_here"
# ```
# 
# **Option 2: Set manually in notebook**
# - The notebook will prompt you to enter the API key when you run the import cell
# - Or uncomment and run the helper cell after imports
# 
# **⚠️ Important Notes:**
# - Replace `"your_actual_api_key_here"` with your real API key
# - Don't share your API key publicly
# - The API key in the example above is just a placeholder
# 
# ### Create Database:
# ```bash
# # Create a database for our RAG system
# createdb multimodal_rag
# psql multimodal_rag -c "CREATE EXTENSION vector;"
# ```

# %%
import os
import json
import base64
import pymupdf
import requests
import logging
import numpy as np
import warnings
import psycopg2
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from IPython import display
import google.generativeai as genai
from pgvector.psycopg2 import register_vector
import tabula
import getpass

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Configure Google Gemini API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    print("⚠️  GOOGLE_API_KEY environment variable not found")
    print("   Get your API key from: https://makersuite.google.com/app/apikey")
    print()
    
    # Offer to set it manually for this session
    response = input("Would you like to enter your Google API key now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        GOOGLE_API_KEY = getpass.getpass("Enter your Google API key: ").strip()
        if GOOGLE_API_KEY:
            # Set for current session
            os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
            print("✅ API key set for this session")
        else:
            print("❌ No API key provided")
    else:
        print("\n💡 To set permanently, run in terminal:")
        print("   export GOOGLE_API_KEY='your_api_key_here'")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("✅ Google Gemini API configured successfully")
        
        # Test the API key by listing available models
        models = list(genai.list_models())
        embedding_models = [m for m in models if 'embedding' in m.name.lower()]
        if embedding_models:
            print(f"✅ Found {len(embedding_models)} embedding model(s) available")
        else:
            print("⚠️  No embedding models found - API key might be invalid")
            
    except Exception as e:
        print(f"❌ Error configuring Google Gemini API: {str(e)}")
        print("   Please check your API key and try again")
        GOOGLE_API_KEY = None
else:
    print("❌ Cannot proceed without Google API key")
    print("   Please set GOOGLE_API_KEY and restart the notebook")

# Database configuration with fallback options
def get_db_config():
    """Get database configuration with user-specific settings"""
    
    # Try different common PostgreSQL user configurations
    possible_users = [
        os.getenv('USER'),  # Current system user
        'postgres',         # Default PostgreSQL user
        os.getenv('DB_USER'),  # Environment variable
    ]
    
    # Filter out None values
    possible_users = [user for user in possible_users if user]
    
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'multimodal_rag'),
        'user': possible_users[0] if possible_users else 'postgres',
        'password': os.getenv('DB_PASSWORD', ''),
        'port': int(os.getenv('DB_PORT', 5432))
    }

DB_CONFIG = get_db_config()
print(f"📝 Database config: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

# Test database connection
def test_db_connection():
    """Test database connection and provide helpful error messages"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        print("✅ Database connection successful")
        return True
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        print(f"❌ Database connection failed: {error_msg}")
        
        if "role" in error_msg and "does not exist" in error_msg:
            print("\n🔧 Quick Fix Options:")
            print("1. Create the database user:")
            print(f"   createuser -s {DB_CONFIG['user']}")
            print("\n2. Or use your system username:")
            print(f"   export DB_USER={os.getenv('USER')}")
            print("\n3. Or create database with your current user:")
            print(f"   createdb -U {os.getenv('USER')} multimodal_rag")
            
        elif "database" in error_msg and "does not exist" in error_msg:
            print("\n🔧 Quick Fix:")
            print(f"   createdb {DB_CONFIG['database']}")
            
        return False
    except Exception as e:
        print(f"❌ Unexpected database error: {str(e)}")
        return False

# Test the connection
test_db_connection()

# %%
# 🔑 Google API Key Configuration Helper
# Run this cell if you need to set your API key manually

def configure_google_api_key():
    """Helper function to configure Google API key"""
    global GOOGLE_API_KEY
    
    current_key = os.getenv('GOOGLE_API_KEY')
    if current_key:
        print(f"✅ Current API key: {current_key[:10]}...{current_key[-4:]}")
        change = input("Do you want to change it? (y/n): ").lower().strip()
        if change not in ['y', 'yes']:
            return current_key
    
    print("\n🔑 Setting up Google API Key")
    print("   1. Go to: https://makersuite.google.com/app/apikey")
    print("   2. Create a new API key")
    print("   3. Copy and paste it below")
    print()
    
    api_key = getpass.getpass("Enter your Google API key: ").strip()
    
    if api_key:
        # Test the API key
        try:
            genai.configure(api_key=api_key)
            # Quick test by listing models
            models = list(genai.list_models())
            if models:
                print("✅ API key is valid!")
                os.environ['GOOGLE_API_KEY'] = api_key
                GOOGLE_API_KEY = api_key
                return api_key
            else:
                print("❌ API key might be invalid - no models returned")
                return None
        except Exception as e:
            print(f"❌ API key test failed: {str(e)}")
            return None
    else:
        print("❌ No API key provided")
        return None

# Uncomment the line below if you need to set/change your API key
configure_google_api_key()

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#    Data Loading
# </h2>

# %%
# Downloading the dataset - URL of the "Attention Is All You Need" paper (Replace it with the URL of the PDF file/dataset you want to download)
url = "https://arxiv.org/pdf/1706.03762.pdf"

# Set the filename and filepath
filename = "attention_paper.pdf"
filepath = os.path.join("data", filename)

# Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download the file
response = requests.get(url)
if response.status_code == 200:
    with open(filepath, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully: {filepath}")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

# %%
# Display the PDF file
display.IFrame(filepath, width=1000, height=600)

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#    Data Extraction
# </h2>

# %%
# Database setup and management
def setup_database():
    """Setup PostgreSQL database with pgvector extension and create tables"""
    try:
        # Test connection first
        if not test_db_connection():
            print("\n🔧 Attempting to fix database connection issues...")
            return setup_database_with_fixes()
        
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        
        # Create extension if not exists
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table for storing embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                page_num INTEGER,
                content_type VARCHAR(20),
                content_text TEXT,
                file_path TEXT,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for faster similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS embedding_idx 
            ON document_embeddings USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database setup completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {str(e)}")
        return False

def setup_database_with_fixes():
    """Attempt to setup database with common fixes"""
    global DB_CONFIG
    
    # Try with current system user
    current_user = os.getenv('USER')
    if current_user and current_user != DB_CONFIG['user']:
        print(f"🔄 Trying with system user: {current_user}")
        DB_CONFIG['user'] = current_user
        
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            print(f"✅ Connected successfully with user: {current_user}")
            return setup_database()
        except:
            pass
    
    # Try with empty password and different user combinations
    for user in ['postgres', current_user, 'admin']:
        if not user:
            continue
            
        test_config = DB_CONFIG.copy()
        test_config['user'] = user
        test_config['password'] = ''
        
        try:
            conn = psycopg2.connect(**test_config)
            conn.close()
            print(f"✅ Connected successfully with user: {user}")
            DB_CONFIG.update(test_config)
            return setup_database()
        except:
            continue
    
    print("❌ Could not establish database connection with any configuration")
    print("\n🛠️  Manual Setup Required:")
    print("1. Create PostgreSQL user and database:")
    print(f"   sudo -u postgres createuser -s {os.getenv('USER')}")
    print(f"   sudo -u postgres createdb -O {os.getenv('USER')} multimodal_rag")
    print("   sudo -u postgres psql multimodal_rag -c 'CREATE EXTENSION vector;'")
    print("\n2. Or use existing PostgreSQL user:")
    print("   Set environment variables: DB_USER, DB_PASSWORD, DB_HOST, DB_PORT")
    
    return False

def clear_database():
    """Clear all embeddings from the database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_embeddings;")
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing database: {str(e)}")

# Create the directories
def create_directories(base_dir):
    directories = ["images", "text", "tables", "page_images"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)

# Check if Java is available for tabula
def check_java_available():
    try:
        import subprocess
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Process tables with fallback
def process_tables(doc, page_num, base_dir, items):
    try:
        # First check if Java is available
        if not check_java_available():
            return
            
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
            table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
            with open(table_file_name, 'w') as f:
                f.write(table_text)
            items.append({"page": page_num, "type": "table", "text": table_text, "path": table_file_name})
    except Exception as e:
        print(f"Error extracting tables from page {page_num}: {str(e)}")

# Alternative table detection using text patterns (fallback method)
def detect_tables_from_text(text, page_num, base_dir, items):
    """Simple fallback method to detect table-like structures in text"""
    lines = text.split('\n')
    potential_tables = []
    current_table = []
    
    for line in lines:
        # Look for lines that might be table rows (contain multiple spaces or tabs)
        if len(line.split()) > 3 and ('  ' in line or '\t' in line):
            current_table.append(line.strip())
        else:
            if len(current_table) > 2:  # At least 3 rows to consider it a table
                potential_tables.append('\n'.join(current_table))
            current_table = []
    
    # Don't forget the last table
    if len(current_table) > 2:
        potential_tables.append('\n'.join(current_table))
    
    # Save detected tables
    for table_idx, table_text in enumerate(potential_tables):
        table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_text_table_{page_num}_{table_idx}.txt"
        with open(table_file_name, 'w') as f:
            f.write(table_text)
        items.append({"page": page_num, "type": "table", "text": table_text, "path": table_file_name})

# Process text chunks
def process_text_chunks(text, text_splitter, page_num, base_dir, items):
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{base_dir}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w') as f:
            f.write(chunk)
        items.append({"page": page_num, "type": "text", "text": chunk, "path": text_file_name})

# Process images
def process_images(page, page_num, base_dir, items):
    images = page.get_images()
    for idx, image in enumerate(images):
        xref = image[0]
        pix = pymupdf.Pixmap(doc, xref)
        image_name = f"{base_dir}/images/{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png"
        pix.save(image_name)
        with open(image_name, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf8')
        items.append({"page": page_num, "type": "image", "path": image_name, "image": encoded_image})

# Process page images
def process_page_images(page, page_num, base_dir, items):
    pix = page.get_pixmap()
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})

# %%
# Initialize database
if not setup_database():
    raise Exception("Failed to setup database. Please check your PostgreSQL installation and configuration.")

# Clear previous data
clear_database()

# Process PDF
doc = pymupdf.open(filepath)
num_pages = len(doc)
base_dir = "data"

# Creating the directories
create_directories(base_dir)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
items = []

# Check if Java is available for table extraction
java_available = check_java_available()
if not java_available:
    print("⚠️  Java not detected. Table extraction with tabula will be skipped.")
    print("   Using fallback text-based table detection instead.")
    print("   To get better table extraction, install Java from https://www.java.com")
    print()

# Process each page of the PDF
for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
    page = doc[page_num]
    text = page.get_text()
    
    # Try tabula table extraction if Java is available, otherwise use fallback
    if java_available:
        process_tables(doc, page_num, base_dir, items)
    else:
        # Use fallback text-based table detection
        detect_tables_from_text(text, page_num, base_dir, items)
    
    process_text_chunks(text, text_splitter, page_num, base_dir, items)
    process_images(page, page_num, base_dir, items)
    process_page_images(page, page_num, base_dir, items)

print(f"\n✅ Processed {len(items)} items from {num_pages} pages")

# %%
# Looking at the first text item
[i for i in items if i['type'] == 'text'][0]

# %%
# Looking at table items
table_items = [i for i in items if i['type'] == 'table']

if table_items:
    print("📊 First table item found:")
    print(table_items[0])
else:
    print("❌ No table items found.")
    print("   This is expected when Java is not available for tabula table extraction.")
    print("   The fallback text-based detection didn't find any table-like structures.")
    
    # Show what content types were actually found
    content_types = {}
    for item in items:
        content_type = item['type']
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    print(f"\n📋 Content types found:")
    for content_type, count in content_types.items():
        print(f"   {content_type}: {count} items")
    
    # Show a sample table-like text chunk if available
    print(f"\n💡 Sample text chunk (might contain table data):")
    if items:
        sample_item = items[0]
        print(f"   Type: {sample_item['type']}")
        print(f"   Page: {sample_item['page']}")
        print(f"   Preview: {sample_item['text'][:200]}...")

# %%
# Looking at image items
image_items = [i for i in items if i['type'] == 'image']

if image_items:
    print("🖼️ First image item found:")
    # Show details without the large base64 image data
    first_image = image_items[0].copy()
    if 'image' in first_image:
        first_image['image'] = f"[Base64 image data - {len(first_image['image'])} characters]"
    print(first_image)
    print(f"\nTotal images found: {len(image_items)}")
else:
    print("❌ No image items found in the PDF.")
    
# Also check page images
page_items = [i for i in items if i['type'] == 'page']
if page_items:
    print(f"\n📄 Page images found: {len(page_items)}")
    print("   First page image details:")
    first_page = page_items[0].copy()
    if 'image' in first_page:
        first_page['image'] = f"[Base64 image data - {len(first_page['image'])} characters]"
    print(first_page)
else:
    print("\n❌ No page images found.")

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#   Generating Multimodal Embeddings with Google Gemini
# </h2>

# %%
# Generate embeddings using Google Gemini
def generate_gemini_embeddings(text=None, image_data=None):
    """
    Generate embeddings using Google Gemini model with payload size checking
    
    Args:
        text (str): Text content to embed
        image_data (str): Base64 encoded image data
    
    Returns:
        list: Embedding vector (768 dimensions)
    """
    try:
        # Check payload size limits
        MAX_PAYLOAD_SIZE = 32000  # Keep under 36KB limit with some margin
        
        if text:
            # Check text size - if too large, truncate it
            if len(text.encode('utf-8')) > MAX_PAYLOAD_SIZE:
                print(f"⚠️  Text too large ({len(text.encode('utf-8'))} bytes), truncating...")
                # Truncate to fit within limit (accounting for encoding overhead)
                text = text[:MAX_PAYLOAD_SIZE // 2]  # Conservative truncation
        
        if image_data:
            # Check image size - if too large, skip it
            image_bytes_size = len(base64.b64decode(image_data))
            if image_bytes_size > MAX_PAYLOAD_SIZE:
                print(f"⚠️  Image too large ({image_bytes_size} bytes), skipping...")
                return None
        
        if text and image_data:
            # For multimodal content (text + image)
            image_bytes = base64.b64decode(image_data)
            response = genai.embed_content(
                model="models/embedding-001",
                content=[text, {"mime_type": "image/png", "data": image_bytes}],
                task_type="RETRIEVAL_DOCUMENT"
            )
        elif text:
            # For text-only content
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
        elif image_data:
            # For image-only content
            image_bytes = base64.b64decode(image_data)
            response = genai.embed_content(
                model="models/embedding-001",
                content={"mime_type": "image/png", "data": image_bytes},
                task_type="RETRIEVAL_DOCUMENT"
            )
        else:
            raise ValueError("Either text or image_data must be provided")
        
        return response['embedding']
    
    except Exception as e:
        error_msg = str(e)
        if "payload size exceeds" in error_msg:
            print(f"⚠️  Payload too large, skipping this item")
        else:
            print(f"Error generating embedding: {error_msg}")
        return None

def chunk_large_text(text, max_chunk_size=20000):
    """
    Split large text into smaller chunks for embedding
    
    Args:
        text (str): Text to chunk
        max_chunk_size (int): Maximum size per chunk in bytes
    
    Returns:
        list: List of text chunks
    """
    if len(text.encode('utf-8')) <= max_chunk_size:
        return [text]
    
    # Split text into sentences first, then combine into chunks
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        if len(test_chunk.encode('utf-8')) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # Single sentence is too large, truncate it
                chunks.append(sentence[:max_chunk_size//2])
                current_chunk = ""
        else:
            current_chunk = test_chunk
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def store_embedding_in_db(page_num, content_type, content_text, file_path, embedding):
    """Store embedding in PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO document_embeddings 
            (page_num, content_type, content_text, file_path, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (page_num, content_type, content_text, file_path, embedding))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error storing embedding: {str(e)}")
        return False

# %%
# Count the number of each type of item
item_counts = {
    'text': sum(1 for item in items if item['type'] == 'text'),
    'table': sum(1 for item in items if item['type'] == 'table'),
    'image': sum(1 for item in items if item['type'] == 'image'),
    'page': sum(1 for item in items if item['type'] == 'page')
}

print(f"Found {item_counts['text']} text chunks, {item_counts['table']} tables, {item_counts['image']} images, {item_counts['page']} page images")

# Initialize counters
counters = dict.fromkeys(item_counts.keys(), 0)
stored_count = 0
skipped_count = 0

# Generate embeddings for all items and store in database
with tqdm(
    total=len(items),
    desc="Generating embeddings and storing in database",
    bar_format=(
        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
) as pbar:
    
    for item in items:
        item_type = item['type']
        counters[item_type] += 1
        
        # Generate embeddings based on content type
        if item_type in ['text', 'table']:
            # For text or table, check size and chunk if necessary
            text_content = item['text']
            
            # Check if text is too large
            if len(text_content.encode('utf-8')) > 32000:
                # Split large text into chunks
                text_chunks = chunk_large_text(text_content, max_chunk_size=20000)
                
                # Process each chunk separately
                for chunk_idx, chunk in enumerate(text_chunks):
                    embedding = generate_gemini_embeddings(text=chunk)
                    
                    if embedding:
                        # Add chunk index to content text for identification
                        chunk_content_text = f"{chunk} [Chunk {chunk_idx+1}/{len(text_chunks)}]"
                        success = store_embedding_in_db(
                            page_num=item['page'],
                            content_type=f"{item_type}_chunk",
                            content_text=chunk_content_text,
                            file_path=f"{item['path']}_chunk_{chunk_idx}",
                            embedding=embedding
                        )
                        if success:
                            stored_count += 1
                    else:
                        skipped_count += 1
            else:
                # Regular size text
                embedding = generate_gemini_embeddings(text=text_content)
                content_text = text_content
                
                if embedding:
                    success = store_embedding_in_db(
                        page_num=item['page'],
                        content_type=item_type,
                        content_text=content_text,
                        file_path=item['path'],
                        embedding=embedding
                    )
                    if success:
                        stored_count += 1
                        # Store embedding in item for local use too
                        item['embedding'] = embedding
                else:
                    skipped_count += 1
                    
        else:
            # For images, check size before processing
            try:
                image_size = len(base64.b64decode(item['image']))
                if image_size > 32000:
                    print(f"⚠️  Skipping large image ({image_size} bytes) from page {item['page']}")
                    skipped_count += 1
                else:
                    embedding = generate_gemini_embeddings(image_data=item['image'])
                    content_text = f"Image from page {item['page']}"
                    
                    if embedding:
                        success = store_embedding_in_db(
                            page_num=item['page'],
                            content_type=item_type,
                            content_text=content_text,
                            file_path=item['path'],
                            embedding=embedding
                        )
                        if success:
                            stored_count += 1
                            # Store embedding in item for local use too
                            item['embedding'] = embedding
                    else:
                        skipped_count += 1
            except Exception as e:
                print(f"⚠️  Error processing image from page {item['page']}: {str(e)}")
                skipped_count += 1
        
        # Update the progress bar
        pbar.set_postfix_str(f"Text: {counters['text']}/{item_counts['text']}, Tables: {counters['table']}/{item_counts['table']}, Images: {counters['image']}/{item_counts['image']}, Stored: {stored_count}, Skipped: {skipped_count}")
        pbar.update(1)

print(f"\n✅ Successfully generated and stored {stored_count} embeddings in PostgreSQL database")
if skipped_count > 0:
    print(f"⚠️  Skipped {skipped_count} items due to size limits or errors")

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#   PostgreSQL Vector Database Operations
# </h2>

# %%
def search_similar_embeddings(query_embedding, top_k=5):
    """
    Search for similar embeddings in PostgreSQL database
    
    Args:
        query_embedding (list): Query embedding vector
        top_k (int): Number of top results to return
    
    Returns:
        list: List of similar items with similarity scores
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        
        # Use cosine similarity for search
        cursor.execute("""
            SELECT id, page_num, content_type, content_text, file_path,
                   1 - (embedding <=> %s::vector) as similarity
            FROM document_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result[0],
                'page': result[1],
                'type': result[2],
                'text': result[3],
                'path': result[4],
                'similarity': result[5]
            })
        
        return formatted_results
        
    except Exception as e:
        print(f"Error searching embeddings: {str(e)}")
        return []

def get_database_stats():
    """Get statistics about the database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content_type, COUNT(*) 
            FROM document_embeddings 
            GROUP BY content_type
        """)
        
        stats = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print("📊 Database Statistics:")
        total = 0
        for content_type, count in stats:
            print(f"   {content_type}: {count} items")
            total += count
        print(f"   Total: {total} embeddings stored")
        
        return stats
        
    except Exception as e:
        print(f"Error getting database stats: {str(e)}")
        return []

# Get database statistics
get_database_stats()

# %%
def generate_answer_with_gemini(query, matched_items):
    """
    Generate answer using Google Gemini Pro Vision model
    
    Args:
        query (str): User's question
        matched_items (list): Retrieved relevant items
    
    Returns:
        str: Generated answer
    """
    try:
        # Prepare context from matched items
        context_parts = []
        
        # Add text and table content
        text_context = []
        for item in matched_items:
            if item['type'] in ['text', 'table']:
                text_context.append(f"[{item['type'].upper()}] {item['text']}")
        
        if text_context:
            context_parts.append("RELEVANT TEXT CONTENT:\n" + "\n\n".join(text_context))
        
        # Collect images
        images = []
        for item in matched_items:
            if item['type'] in ['image', 'page']:
                # Load image from file
                try:
                    with open(item['path'], 'rb') as f:
                        images.append({
                            "mime_type": "image/png",
                            "data": f.read()
                        })
                except:
                    pass
        
        # Prepare the prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context and images. 
        Use the relevant text content and analyze any provided images to give accurate and comprehensive answers.
        If the information is not available in the context, say so clearly."""
        
        context_text = "\n\n".join(context_parts)
        full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context_text}\n\nQUESTION: {query}\n\nANSWER:"
        
        # Prepare content for Gemini
        content = [full_prompt]
        content.extend(images)
        
        # Generate response using Gemini Pro Vision
        model = genai.GenerativeModel('gemini-pro-vision' if images else 'gemini-pro')
        response = model.generate_content(content)
        
        return response.text
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"

# Function to load image data for items (for backward compatibility)
def load_image_data_for_items(items):
    """Load image data for items that need it"""
    for item in items:
        if item['type'] in ['image', 'page'] and 'image' not in item:
            try:
                with open(item['path'], 'rb') as f:
                    item['image'] = base64.b64encode(f.read()).decode('utf8')
            except:
                pass
    return items

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#   Test the RAG Pipeline
# </h2>

# %%
# User Query
query = "Which optimizer was used when training the models?"

print(f"🔍 Query: {query}")

# Generate embeddings for the query using Gemini
query_embedding = generate_gemini_embeddings(text=query)

if query_embedding:
    # Search for similar embeddings in PostgreSQL
    search_results = search_similar_embeddings(query_embedding, top_k=5)
    
    print(f"\n📋 Found {len(search_results)} relevant results:")
    for i, result in enumerate(search_results, 1):
        print(f"   {i}. [{result['type']}] Page {result['page']} - Similarity: {result['similarity']:.3f}")
        print(f"      Preview: {result['text'][:100]}...")
        print()
else:
    print("❌ Failed to generate query embedding")

# %%
# Display detailed search results
if search_results:
    print("🔎 Detailed Search Results:")
    print("-" * 50)
    
    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. Content Type: {result['type'].upper()}")
        print(f"   Page: {result['page']}")
        print(f"   Similarity Score: {result['similarity']:.4f}")
        print(f"   File Path: {result['path']}")
        print(f"   Content Preview: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
        print("-" * 50)
else:
    print("No search results found.")

# %%
# Generate answer using Google Gemini with the retrieved context
if search_results:
    print("🤖 Generating answer with Google Gemini...")
    
    # Convert search results to the format expected by the answer generation function
    matched_items = []
    for result in search_results:
        item = {
            'type': result['type'],
            'text': result['text'],
            'path': result['path'],
            'page': result['page']
        }
        matched_items.append(item)
    
    # Load image data for image items
    matched_items = load_image_data_for_items(matched_items)
    
    # Generate answer
    response = generate_answer_with_gemini(query, matched_items)
    
    print("\n" + "="*60)
    print("📝 ANSWER:")
    print("="*60)
    display.Markdown(response)
    
else:
    print("❌ No relevant content found for the query.")

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#   Your Turn: Test the RAG Pipeline
# </h2>

# %%
# List of queries (Replace with any query of your choice)
other_queries = ["How long were the base and big models trained?",
                 "Which optimizer was used when training the models?",
                 "What is the position-wise feed-forward neural network mentioned in the paper?",
                 "What is the BLEU score of the model in English to German translation (EN-DE)?",
                 "How is the scaled-dot-product attention is calculated?",
                 ]


# %%
def test_rag_query(query_text, top_k=5):
    """Test the RAG pipeline with a query"""
    print(f"🔍 Query: {query_text}")
    print("-" * 60)
    
    # Generate query embedding
    query_embedding = generate_gemini_embeddings(text=query_text)
    
    if not query_embedding:
        print("❌ Failed to generate query embedding")
        return
    
    # Search for similar content
    search_results = search_similar_embeddings(query_embedding, top_k=top_k)
    
    if not search_results:
        print("❌ No relevant content found")
        return
    
    print(f"📋 Found {len(search_results)} relevant results")
    
    # Convert to format for answer generation
    matched_items = []
    for result in search_results:
        item = {
            'type': result['type'],
            'text': result['text'],
            'path': result['path'],
            'page': result['page']
        }
        matched_items.append(item)
    
    # Load image data
    matched_items = load_image_data_for_items(matched_items)
    
    # Generate answer
    print("\n🤖 Generating answer...")
    response = generate_answer_with_gemini(query_text, matched_items)
    
    print("\n" + "="*60)
    print("📝 ANSWER:")
    print("="*60)
    display.Markdown(response)
    print("="*60)

# Test with the first query
query = other_queries[0]  # "How long were the base and big models trained?"
test_rag_query(query)

# %% [markdown]
# <h2 style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #1e90ff); 
#             color: white; 
#             padding: 15px; 
#             border-radius: 10px; 
#             text-align: center; 
#             font-family: 'Comic Sans MS', cursive, sans-serif; 
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
#   Thank you!
# </h2>

# %% [markdown]
# ## Complete Setup Guide for Local Multimodal RAG
# 
# This notebook uses a completely local setup with Google Gemini API for embeddings and PostgreSQL for vector storage.
# 
# ### 1. Install PostgreSQL with pgvector
# 
# #### macOS (using Homebrew):
# ```bash
# # Install PostgreSQL
# brew install postgresql
# 
# # Start PostgreSQL service
# brew services start postgresql
# 
# # Install pgvector extension
# git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
# cd pgvector
# make
# make install
# 
# # Create database and user (recommended approach)
# createuser -s $(whoami)  # Create user with your system username
# createdb multimodal_rag
# psql multimodal_rag -c "CREATE EXTENSION vector;"
# ```
# 
# #### Ubuntu/Linux:
# ```bash
# # Install PostgreSQL
# sudo apt update
# sudo apt install postgresql postgresql-contrib postgresql-server-dev-all
# 
# # Install build tools
# sudo apt install build-essential git
# 
# # Install pgvector
# git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
# cd pgvector
# make
# sudo make install
# 
# # Create database and user
# sudo -u postgres createuser -s $(whoami)
# sudo -u postgres createdb -O $(whoami) multimodal_rag
# sudo -u postgres psql multimodal_rag -c "CREATE EXTENSION vector;"
# ```
# 
# ### 2. Troubleshooting PostgreSQL Connection Issues
# 
# If you see "role does not exist" error, try these fixes:
# 
# #### Option 1: Create PostgreSQL user with your system username (Recommended)
# ```bash
# # Create user with superuser privileges
# sudo -u postgres createuser -s $(whoami)
# 
# # Create database
# createdb multimodal_rag
# psql multimodal_rag -c "CREATE EXTENSION vector;"
# ```
# 
# #### Option 2: Use existing postgres user
# ```bash
# # Create database as postgres user
# sudo -u postgres createdb multimodal_rag
# sudo -u postgres psql multimodal_rag -c "CREATE EXTENSION vector;"
# 
# # Then set environment variable
# export DB_USER=postgres
# ```
# 
# #### Option 3: Set custom database configuration
# ```bash
# export DB_USER=your_username
# export DB_PASSWORD=your_password
# export DB_HOST=localhost
# export DB_PORT=5432
# export DB_NAME=multimodal_rag
# ```
# 
# ### 3. Install Python Dependencies
# ```bash
# pip install google-generativeai psycopg2-binary pgvector numpy tqdm requests pymupdf tabula-py
# ```
# 
# ### 4. Get Google API Key
# 1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
# 2. Create a new API key
# 3. Set environment variable:
# ```bash
# export GOOGLE_API_KEY="your_api_key_here"
# ```
# 
# ### 5. Optional: Install Java for Better Table Extraction
# ```bash
# # macOS
# brew install openjdk
# 
# # Ubuntu/Linux
# sudo apt install default-jdk
# ```
# 
# ### 6. Database Configuration
# The notebook automatically detects and uses the best database configuration:
# 1. Uses your system username as PostgreSQL user (most common)
# 2. Falls back to 'postgres' user if needed
# 3. Respects environment variables for custom configuration
# 
# ### Common PostgreSQL Issues and Solutions:
# 
# #### Issue: "role 'postgres' does not exist"
# **Solution**: Create user with your system username (see Option 1 above)
# 
# #### Issue: "database 'multimodal_rag' does not exist"
# **Solution**: 
# ```bash
# createdb multimodal_rag
# ```
# 
# #### Issue: "permission denied"
# **Solution**: Create user with superuser privileges:
# ```bash
# sudo -u postgres createuser -s $(whoami)
# ```
# 
# #### Issue: PostgreSQL not running
# **Solution**:
# ```bash
# # macOS
# brew services start postgresql
# 
# # Ubuntu/Linux
# sudo systemctl start postgresql
# ```
# 
# ### Benefits of This Local Setup:
# - ✅ **Privacy**: Your data never leaves your machine (except for Google API calls)
# - ✅ **Cost-effective**: Only pay for Google API usage, no cloud storage costs
# - ✅ **Scalable**: PostgreSQL can handle large datasets efficiently
# - ✅ **Flexible**: Easy to modify and extend
# - ✅ **Production-ready**: PostgreSQL is enterprise-grade
# 
# ### Performance Tips:
# - Use SSD storage for better PostgreSQL performance
# - Adjust PostgreSQL configuration for your system
# - Consider using connection pooling for production use
# - Monitor Google API usage to control costs

# %%
# Additional utility functions for testing and management

def test_multiple_queries():
    """Test all predefined queries"""
    print("🚀 Testing all predefined queries...")
    print("="*80)
    
    for i, query in enumerate(other_queries, 1):
        print(f"\n📝 Query {i}/{len(other_queries)}")
        test_rag_query(query, top_k=3)
        print("\n" + "="*80)

def get_content_by_type(content_type):
    """Get all content of a specific type from database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT page_num, content_text, file_path
            FROM document_embeddings
            WHERE content_type = %s
            ORDER BY page_num
        """, (content_type,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"📊 Found {len(results)} {content_type} items:")
        for page_num, content_text, file_path in results:
            print(f"  Page {page_num}: {content_text[:100]}...")
        
        return results
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def cleanup_database():
    """Clean up the database (use with caution!)"""
    response = input("⚠️  This will delete all embeddings. Are you sure? (type 'yes' to confirm): ")
    if response.lower() == 'yes':
        clear_database()
        print("✅ Database cleaned up successfully")
    else:
        print("❌ Operation cancelled")

# Uncomment the line below to test all queries
# test_multiple_queries()


