import logging
from llama_index.core.node_parser import HierarchicalNodeParser
# get_node_parser_errors was removed, as we fixed in the previous step
from llama_index.core.schema import Document, BaseNode
from . import config

# Configure logging
log = logging.getLogger(__name__)

def create_hierarchical_nodes(documents: list[Document]) -> list[BaseNode]:
    """
    Segments documents at multiple granularities (Requirement B)
    using the justified HierarchicalNodeParser.
    
    This automatically creates parent-child relationships.
    """
    log.info(f"Starting hierarchical chunking with sizes: {config.CHUNK_SIZES}")
    
    # This is the core parser that meets the multi-granularity requirement
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=config.CHUNK_SIZES,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    all_nodes = []
    
    log.info("Parsing nodes from documents...")
    try:
        all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    except Exception as e:
        log.error(f"An error occurred during node parsing: {e}", exc_info=True)
        # raise e # Optionally re-raise the exception

    log.info(f"Generated {len(all_nodes)} nodes (parents and children).")
    return all_nodes

if __name__ == "__main__":
    # For testing this module directly
    # This import should work when running as a module:
    # python -m data_pipeline.chunking
    from src.document_loader import load_all_documents, deduplicate_documents
    
    logging.basicConfig(level=logging.INFO)
    docs = load_all_documents()
    docs = deduplicate_documents(docs)
    nodes = create_hierarchical_nodes(docs) # 'nodes' is all_nodes
    
    print(f"Total nodes created: {len(nodes)}")
    
    # Inspect a parent and child
    child_nodes = [n for n in nodes if n.parent_node is not None]
    parent_nodes = [n for n in nodes if n.parent_node is None]

    print(f"Total Parent nodes (chunk_size={config.CHUNK_SIZES[0]}): {len(parent_nodes)}")
    print(f"Total Child nodes (chunk_size={config.CHUNK_SIZES[1]}): {len(child_nodes)}")
    
    if child_nodes:
        print("\n--- Sample Child Node (for embedding) ---")
        print(f"ID: {child_nodes[0].id_}")
        
        # *** CORRECTION 1: Use .node_id to get the ID from RelatedNodeInfo ***
        parent_id = child_nodes[0].parent_node.node_id
        print(f"Parent ID: {parent_id}")

        print("\n--- Its Parent Node (for context) ---")
        
        # *** CORRECTION 2: Find the actual parent node from the 'nodes' list ***
        parent_node_obj = next((n for n in nodes if n.id_ == parent_id), None)
        
        if parent_node_obj:
            print(f"ID: {parent_node_obj.id_}")
            print(f"Text: {parent_node_obj.text[:100]}...")
        else:
            print(f"Parent node with ID {parent_id} not found in the main list.")
    
    elif not docs:
         print("\nNo documents were loaded. Skipping node inspection.")
    else:
        print("\nDocuments were loaded, but no child nodes were created.")