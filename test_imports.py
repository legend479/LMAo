#!/usr/bin/env python3
"""
Test script to verify all __init__.py imports work correctly
"""


def test_main_module():
    """Test main src module imports"""
    try:
        import src

        print("✓ Main src module imported successfully")

        # Test version info
        print(f"  Version: {src.__version__}")
        print(f"  Author: {src.__author__}")
        print(f"  Description: {src.__description__}")

        # Test submodule access
        assert hasattr(src, "shared")
        assert hasattr(src, "rag_pipeline")
        assert hasattr(src, "api_server")
        assert hasattr(src, "agent_server")
        print("  ✓ All submodules accessible")

    except Exception as e:
        print(f"✗ Main module import failed: {e}")
        return False
    return True


def test_shared_module():
    """Test shared module imports"""
    try:
        from src import shared

        print("✓ Shared module imported successfully")

        # Test key classes
        assert hasattr(shared, "get_settings")
        assert hasattr(shared, "get_logger")
        assert hasattr(shared, "ServiceStatus")
        assert hasattr(shared, "HealthCheck")
        print("  ✓ Key shared classes accessible")

    except Exception as e:
        print(f"✗ Shared module import failed: {e}")
        return False
    return True


def test_rag_pipeline_module():
    """Test RAG pipeline module imports"""
    try:
        from src import rag_pipeline

        print("✓ RAG pipeline module imported successfully")

        # Test key classes
        assert hasattr(rag_pipeline, "RAGPipeline")
        assert hasattr(rag_pipeline, "DocumentProcessor")
        assert hasattr(rag_pipeline, "DocumentType")
        print("  ✓ Key RAG pipeline classes accessible")

    except Exception as e:
        print(f"✗ RAG pipeline module import failed: {e}")
        return False
    return True


def test_api_server_module():
    """Test API server module imports"""
    try:
        from src import api_server

        print("✓ API server module imported successfully")

        # Test key classes
        assert hasattr(api_server, "APIServer")
        assert hasattr(api_server, "JWTManager")
        assert hasattr(api_server, "routers")
        print("  ✓ Key API server classes accessible")

    except Exception as e:
        print(f"✗ API server module import failed: {e}")
        return False
    return True


def test_agent_server_module():
    """Test agent server module imports"""
    try:
        from src import agent_server

        print("✓ Agent server module imported successfully")

        # Test key classes
        assert hasattr(agent_server, "LangGraphOrchestrator")
        assert hasattr(agent_server, "tools")
        assert hasattr(agent_server, "KnowledgeRetrievalTool")
        print("  ✓ Key agent server classes accessible")

    except Exception as e:
        print(f"✗ Agent server module import failed: {e}")
        return False
    return True


def test_llm_module():
    """Test LLM module imports"""
    try:
        from src.shared import llm

        print("✓ LLM module imported successfully")

        # Test key classes
        assert hasattr(llm, "LLMClient")
        assert hasattr(llm, "OpenAIProvider")
        assert hasattr(llm, "create_llm_client")
        print("  ✓ Key LLM classes accessible")

    except Exception as e:
        print(f"✗ LLM module import failed: {e}")
        return False
    return True


def test_database_module():
    """Test database module imports"""
    try:
        from src.shared import database

        print("✓ Database module imported successfully")

        # Test key classes
        assert hasattr(database, "Base")
        assert hasattr(database, "User")
        assert hasattr(database, "DatabaseManager")
        print("  ✓ Key database classes accessible")

    except Exception as e:
        print(f"✗ Database module import failed: {e}")
        return False
    return True


def main():
    """Run all import tests"""
    print("Testing __init__.py imports...\n")

    tests = [
        test_main_module,
        test_shared_module,
        test_rag_pipeline_module,
        test_api_server_module,
        test_agent_server_module,
        test_llm_module,
        test_database_module,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All __init__.py files are properly configured!")
        return True
    else:
        print("❌ Some imports failed. Check the error messages above.")
        return False


if __name__ == "__main__":
    main()
