#!/usr/bin/env python3
"""
Final test to verify RAG Pipeline fixes are working
"""


def test_rag_pipeline_import():
    """Test that RAG pipeline can be imported without circular import errors"""
    try:
        from src.rag_pipeline.main import RAGPipeline

        print("✅ RAG Pipeline imported successfully")

        # Test instantiation
        pipeline = RAGPipeline()
        print("✅ RAG Pipeline instantiated successfully")

        # Test initial state
        assert not pipeline._initialized
        assert pipeline.document_processor is None
        print("✅ Pipeline shows correct initial state")

        return True
    except Exception as e:
        print(f"❌ RAG Pipeline import/instantiation failed: {e}")
        return False


def test_component_imports():
    """Test that all RAG components can be imported"""
    try:
        from src.rag_pipeline.document_processor import DocumentProcessor
        from src.rag_pipeline.document_ingestion import DocumentIngestionService
        from src.rag_pipeline.vector_store import ElasticsearchStore
        from src.rag_pipeline.search_engine import HybridSearchEngine
        from src.rag_pipeline.embedding_manager import EmbeddingManager
        from src.rag_pipeline.chunking_strategies import ChunkingManager

        print("✅ All RAG components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Component import failed: {e}")
        return False


def test_chunking_strategies_fix():
    """Test that chunking strategies circular import is resolved"""
    try:
        from src.rag_pipeline.chunking_strategies import (
            ChunkingManager,
            ChunkingConfigAdvanced,
        )
        from src.rag_pipeline.models import ChunkingConfig

        # Test that both config types work
        basic_config = ChunkingConfig(strategy="hierarchical")
        advanced_config = ChunkingConfigAdvanced()

        print("✅ Chunking strategies and configs work correctly")
        return True
    except Exception as e:
        print(f"❌ Chunking strategies test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing RAG Pipeline fixes...")

    tests = [
        test_rag_pipeline_import,
        test_component_imports,
        test_chunking_strategies_fix,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All RAG Pipeline dependency fixes are working correctly!")
    else:
        print("⚠️  Some issues remain to be fixed")
