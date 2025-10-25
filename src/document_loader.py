import os
import logging
from datetime import datetime
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from . import config

# Configure logging
log = logging.getLogger(__name__)

def get_source_metadata(file_path: Path) -> dict:
    """
    Generates metadata based on the file's path and system info.
    This fulfills the metadata requirement.
    """
    # Infer 'subject' from the parent directory
    subject = file_path.parent.name
    
    # Use file path as 'source'
    source = str(file_path)
    
    # Get last modified time as 'timestamp'
    timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    
    return {
        "subject": subject,
        "source": source,
        "context": f"Content from file: {file_path.name}",
        "timestamp": timestamp
    }

def load_all_documents() -> list[Document]:
    """
    Loads all documents from the root directory.
    - Automatically detects file types (PDF, DOCX, etc.)
    - Associates metadata with each document
    """
    log.info(f"Starting document load from: {config.DOCS_DIR}")
    
    # This reader handles all specified formats (PDF, DOCX, PPTX, TXT, MD)
    # It uses 'unstructured' library in the background.
    reader = SimpleDirectoryReader(
        input_dir=config.DOCS_DIR,
        required_exts=config.REQUIRED_EXTS,
        recursive=True
    )
    
    loaded_docs = []
    
    # Load data and enhance metadata
    for doc in reader.load_data():
        # 'load_data' provides some metadata, we'll enhance it.
        file_path = Path(doc.metadata.get('file_path'))
        custom_metadata = get_source_metadata(file_path)
        
        # Combine default and custom metadata
        doc.metadata.update(custom_metadata)
        
        # --- B. Preprocessing (Partial) ---
        # Apply lowercasing and basic cleaning
        
        # *** THIS IS THE CORRECTED LINE ***
        doc.set_content(doc.text.lower())
        
        # Add more cleaning rules here (e.g., regex for non-informative content)
        
        loaded_docs.append(doc)
        
    log.info(f"Successfully loaded {len(loaded_docs)} documents.")
    return loaded_docs

def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """
    Implements deduplication based on a hash of the document text.
    """
    log.info(f"Starting deduplication for {len(documents)} documents...")
    seen_hashes = set()
    deduped_docs = []
    
    for doc in documents:
        doc_hash = hash(doc.text)
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            deduped_docs.append(doc)
            
    removed_count = len(documents) - len(deduped_docs)
    log.info(f"Removed {removed_count} duplicate documents.")
    return deduped_docs

if __name__ == "__main__":
    # For testing this module directly
    # Need to fix the relative import for direct execution
    from . import config
    
    logging.basicConfig(level=logging.INFO)
    docs = load_all_documents()
    docs = deduplicate_documents(docs)
    print(f"Loaded and deduplicated {len(docs)} documents.")
    
    if docs:
        print("\nSample Document Metadata:")
        print(docs[0].metadata)
    else:
        print("\nNo documents found or loaded.")