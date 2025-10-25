import logging
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.schema import BaseNode
# CORRECTED IMPORT for Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
# CORRECTED IMPORT for Reranker
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from . import config

# Configure logging
log = logging.getLogger(__name__)

def get_embedding_model(model_name: str) -> HuggingFaceEmbedding:
    """
    Initializes the embedding model.
    """
    log.info(f"Loading embedding model: {model_name}")
    # Using 'trust_remote_code=True' for BGE models
    return HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=config.EMBED_CACHE_DIR,  # <--- THIS IS THE FIX
        trust_remote_code=True,
        parallel_process=False
    )

def create_vector_store() -> tuple[ChromaVectorStore, StorageContext]:
    """
    Initializes the persistent ChromaDB vector store.
    """
    log.info(f"Initializing vector store at: {config.VECTOR_DB_PATH}")
    db = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_context

def build_index(nodes: list[BaseNode], embed_model: HuggingFaceEmbedding, storage_context: StorageContext):
    """
    Builds the VectorStoreIndex.
    This step performs the embedding and indexing.
    """
    log.info("Setting global embedding model...")
    Settings.embed_model = embed_model
    Settings.llm = None  # We are only doing data prep, not querying
    
    log.info(f"Building index with {len(nodes)} nodes...")
    # This builds the index and stores it in the ChromaDB 
    # specified in the storage_context
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    log.info("Index built and stored successfully.")
    return index

def run_indexing_pipeline(nodes: list[BaseNode]):
    """
    Main function to run the full indexing process.
    """
    # Use the recommended domain-specific model
    embed_model = get_embedding_model(config.EMBED_MODEL_DOMAIN)
    
    # (Optional) To use the baseline model instead, uncomment below:
    # embed_model = get_embedding_model(config.EMBED_MODEL_BASE)
    
    _, storage_context = create_vector_store()
    
    index = build_index(nodes, embed_model, storage_context)
    return index

def test_retrieval_with_reranker(index: VectorStoreIndex):
    """
    [BONUS] Demonstrates retrieval with a reranker.
    """
    log.info("Testing retrieval with BGE Reranker [BONUS]...")
    
    # 1. Set up the BGE Reranker
    # CORRECTED CLASS NAME
    reranker = FlagEmbeddingReranker(
        model=config.RERANK_MODEL,
        top_n=config.RERANK_TOP_N
    )
    
    # 2. Set up the base retriever
    # We retrieve more (e.g., top 10) to give the reranker options
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )
    
    # 3. Create the query engine with the reranker
    query_engine = RetrieverQueryEngine.from_args(
        retriever=base_retriever,
        node_postprocessors=[reranker]
    )
    
    # 4. Run a test query
    query = "What is a multi-head attention mechanism?"
    response = query_engine.query(query)
    
    print(f"\n--- Test Query: {query} ---")
    print(f"Found {len(response.source_nodes)} relevant nodes (after reranking).")
    
    for i, node in enumerate(response.source_nodes):
        print(f"\n--- Result {i+1} (Score: {node.score:.4f}) ---")
        print(f"Source: {node.metadata.get('source')}")
        print(f"Text: {node.text[:250]}...")
        # This shows the parent-child link in action!
        if node.parent_node:
            print(f"Context (from Parent Node): {node.parent_node.text[:250]}...")

if __name__ == "__main__":
    from . import config
    # For testing this module directly
    from src.document_loader import load_all_documents, deduplicate_documents
    from src.chunking import create_hierarchical_nodes
    
    logging.basicConfig(level=logging.INFO)
    
    log.info("--- (1/3) Loading Documents ---")
    docs = load_all_documents()
    docs = deduplicate_documents(docs)
    
    log.info("--- (2/3) Creating Nodes ---")
    nodes = create_hierarchical_nodes(docs)
    
    log.info("--- (3/3) Indexing Nodes ---")
    # This will use the domain-specific BGE model
    index = run_indexing_pipeline(nodes)
    
    # Test the bonus reranker
    # test_retrieval_with_reranker(index)