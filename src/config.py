import logging
import os

# --- A. Document Collection ---
# Root directory for all source documents
DOCS_DIR = "./test_doc"

# File patterns to include
# Using SimpleDirectoryReader, this supports:
# .pdf, .docx, .pptx, .txt, .md, .csv, .html, .json, and more
REQUIRED_EXTS = [".pdf", ".docx", ".pptx", ".txt", ".md"]

# --- B. Preprocessing & Chunking ---
# Hierarchical chunk sizes
# 1st level: Parent nodes (broad context)
# 2nd level: Child nodes (for embedding)
# 3rd level: (Optional) Sub-child nodes
CHUNK_SIZES = [2048, 512, 128]

# Overlap between chunks
CHUNK_OVERLAP = 20

# --- C. Embedding & Indexing ---
# Path to the persistent vector database
VECTOR_DB_PATH = "./vector_db_store"
# Collection name in ChromaDB
COLLECTION_NAME = "sme_agent_collection"

# Baseline embedding model
EMBED_MODEL_BASE = "sentence-transformers/all-mpnet-base-v2"

# Domain-specific embedding model (Recommended)
EMBED_MODEL_DOMAIN = "BAAI/bge-large-en-v1.5"
EMBED_CACHE_DIR = os.path.expanduser("~/.cache/llama_index_embeddings")

# [BONUS] Reranker model
RERANK_MODEL = "BAAI/bge-reranker-large"
RERANK_TOP_N = 3

# --- Logging (for Bonus Pipeline) ---
LOG_FILE = "./logs/ingestion.log"
LOG_LEVEL = logging.INFO