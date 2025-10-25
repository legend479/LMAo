import argparse
import logging
import sys
import textwrap
import chromadb

# Import the config from the data_pipeline package
from src import config

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    QueryBundle
)
# We no longer need the retriever
# from llama_index.core.retrievers import VectorIndexRetriever 

# --- THIS IS THE FIX ---
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode
# --- END OF FIX ---

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import NodeWithScore, TextNode

# --- Configuration ---
# Suppress noisy logs
logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
log = logging.getLogger("SME_Query")


def get_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    """
    Initializes and loads the embedding model from cache.
    Applies the patches we discovered.
    """
    log.info(f"Loading embedding model: {model_name}...")
    return HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=config.EMBED_CACHE_DIR,
        trust_remote_code=True,
        parallel_process=False  # Bug fix 1
    )

def load_vector_store() -> ChromaVectorStore:
    """
    Connects to the existing persistent ChromaDB.
    """
    log.info(f"Connecting to existing vector store at: {config.VECTOR_DB_PATH}")
    db = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    chroma_collection = db.get_collection(config.COLLECTION_NAME)
    
    # We apply the bug fix by calling the constructor
    # (assuming the package file is patched)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def print_result(num: int, node: NodeWithScore):
    """Helper function to print a single reranked node."""
    print(f"\n--- Result {num} (Score: {node.score:.4f}) ---")
    print(f"Source:  {node.metadata.get('source')}")
    print(f"Subject: {node.metadata.get('subject')}")
    print("\nText:")
    
    # Wrap the text for readable terminal output
    wrapped_text = textwrap.fill(
        node.text, 
        width=90, 
        initial_indent="    ", 
        subsequent_indent="    "
    )
    print(wrapped_text)
    
    print("-" * 94)

def main(query_text: str):
    """
    Main query pipeline.
    """
    log.info("--- Starting SME Agent Query ---")
    
    # 1. Setup Embed Model (globally)
    embed_model = get_embedding_model(config.EMBED_MODEL_DOMAIN)
    Settings.embed_model = embed_model
    Settings.llm = None  # We are only doing retrieval, not synthesis
    
    # 2. Load the existing vector store
    try:
        vector_store = load_vector_store()
    except Exception as e:
        log.error(f"Failed to load vector store from {config.VECTOR_DB_PATH}")
        log.error("Did you run the ingestion pipeline first?")
        log.error(f"Details: {e}")
        sys.exit(1)
        
    # 3. Setup the BGE Reranker (Bonus)
    log.info(f"Loading reranker model: {config.RERANK_MODEL}...")
    reranker = FlagEmbeddingReranker(
        model=config.RERANK_MODEL,
        top_n=config.RERANK_TOP_N  # Show top 3
    )
    
    # 4. === MANUAL RETRIEVAL (BYPASSES BUGGY RETRIEVER) ===
    log.info(f"Generating embedding for query: '{query_text}'")
    query_bundle = QueryBundle(query_text)
    query_embedding = embed_model.get_query_embedding(query_text)

    # 5. Create a manual VectorStoreQuery
    # We retrieve 10 to give the reranker options
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=10,
        mode=VectorStoreQueryMode.DEFAULT,
    )

    # 6. Query the vector store *directly*
    # This call does not pass the buggy empty filter
    log.info("Querying vector store directly...")
    query_result = vector_store.query(vector_store_query)

    # 7. Manually reconstruct NodeWithScore objects
    # This is what the retriever would have done
    retrieved_nodes = []
    if query_result.nodes:
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            retrieved_nodes.append(NodeWithScore(node=node, score=similarity))
    
    log.info(f"Found {len(retrieved_nodes)} initial nodes.")
    # === END OF MANUAL RETRIEVAL ===

    
    # 8. Rerank the manually retrieved nodes
    log.info("Reranking retrieved nodes...")
    reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
    
    # 9. Print the results
    log.info("--- Query Complete ---")
    if not reranked_nodes:
        log.warning("No relevant documents found for your query.")
        return

    print(f"\nFound {len(reranked_nodes)} relevant results (after reranking):")
    for i, node in enumerate(reranked_nodes):
        print_result(i + 1, node)

def setup_cli():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Query the SME Agent's knowledge base.")
    parser.add_argument(
        "query",
        type=str,
        nargs='+',  # Allows the query to be multiple words
        help="The query text to search for."
    )
    return parser

if __name__ == "__main__":
    # 1. Setup CLI
    parser = setup_cli()
    args = parser.parse_args()
    
    # 2. Combine all CLI args into a single query string
    query_text = " ".join(args.query)
    
    # 3. Run the main query function
    try:
        main(query_text)
    except Exception as e:
        log.error(f"An error occurred during query: {e}", exc_info=True)
        sys.exit(1)