#!/usr/bin/env python3
"""
Diagnostic script to identify performance test failures
Run this before running the full test suite
"""

import asyncio
import httpx
import os
from pathlib import Path


async def check_agent_service():
    """Check if Agent service is reachable"""
    print("\n🔍 Checking Agent Service...")

    # Get config from environment
    agent_host = os.getenv("AGENT_HOST", "localhost")
    agent_port = os.getenv("AGENT_PORT", "8001")
    agent_url = f"http://{agent_host}:{agent_port}"

    print(f"   URL: {agent_url}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{agent_url}/health")
            if response.status_code == 200:
                print(f"   ✅ Agent service is UP")
                print(f"   Response: {response.json()}")
                return True
            else:
                print(f"   ❌ Agent service returned {response.status_code}")
                return False
    except httpx.ConnectError:
        print(f"   ❌ Cannot connect to Agent service")
        print(f"   💡 Start it with: python -m src.agent_server.main")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def check_rag_service():
    """Check if RAG service is reachable"""
    print("\n🔍 Checking RAG Service...")

    rag_host = os.getenv("RAG_HOST", "localhost")
    rag_port = os.getenv("RAG_PORT", "8002")
    rag_url = f"http://{rag_host}:{rag_port}"

    print(f"   URL: {rag_url}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{rag_url}/health")
            if response.status_code == 200:
                print(f"   ✅ RAG service is UP")
                print(f"   Response: {response.json()}")
                return True
            else:
                print(f"   ⚠️  RAG service returned {response.status_code}")
                return False
    except httpx.ConnectError:
        print(f"   ⚠️  Cannot connect to RAG service")
        print(f"   💡 Start it with: python -m src.rag_pipeline.main")
        return False
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        return False


def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking Environment Configuration...")

    required_vars = [
        "AGENT_HOST",
        "AGENT_PORT",
        "RAG_HOST",
        "RAG_PORT",
        "SECRET_KEY",
    ]

    optional_vars = [
        "DATABASE_URL",
        "UPLOAD_DIR",
        "MAX_FILE_SIZE",
    ]

    print("\n   Required variables:")
    all_present = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "SECRET" in var:
                display_value = value[:4] + "..." if len(value) > 4 else "***"
            else:
                display_value = value
            print(f"   ✅ {var}={display_value}")
        else:
            print(f"   ❌ {var} is not set")
            all_present = False

    print("\n   Optional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}={value}")
        else:
            print(f"   ⚠️  {var} is not set (using default)")

    return all_present


def check_directories():
    """Check required directories exist"""
    print("\n🔍 Checking Directories...")

    upload_dir = os.getenv("UPLOAD_DIR", "uploads")
    temp_dir = os.getenv("TEMP_DIR", "temp")

    dirs_to_check = [
        ("Upload directory", upload_dir),
        ("Temp directory", temp_dir),
        ("Logs directory", "logs"),
    ]

    all_exist = True
    for name, path in dirs_to_check:
        if Path(path).exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ⚠️  {name} does not exist: {path}")
            print(f"      Will be created automatically")

    return all_exist


async def test_document_upload_payload():
    """Test if document upload payload is valid"""
    print("\n🔍 Testing Document Upload Payload Format...")

    # This would require the API server to be running
    # For now, just show what the test is sending

    print("   Test sends:")
    print("   - File: doc_0.txt (text/plain)")
    print("   - Content: 'Load test document 0\\n' + 100 lines")
    print("   - Metadata: '{\"doc_id\": 0}'")
    print("   - Requires: Authentication (mocked in test)")
    print("   💡 Check src/api_server/routers/documents.py for validation rules")


def check_validation_limits():
    """Show validation middleware limits"""
    print("\n🔍 Validation Middleware Limits...")

    print("   From src/api_server/middleware/validation.py:")
    print("   - MAX_REQUEST_SIZE: 10 MB")
    print("   - MAX_JSON_DEPTH: 10")
    print("   - MAX_ARRAY_LENGTH: 1000")
    print("   - MAX_STRING_LENGTH: 100,000 bytes (100 KB)")
    print("")
    print("   💡 Large payload test sends up to 150KB")
    print("   💡 Payloads >100KB should be rejected with HTTP 400")


async def main():
    """Run all diagnostics"""
    print("=" * 60)
    print("Performance Test Suite Diagnostics")
    print("=" * 60)

    # Check environment
    env_ok = check_environment()

    # Check directories
    dirs_ok = check_directories()

    # Check services
    agent_ok = await check_agent_service()
    rag_ok = await check_rag_service()

    # Show validation info
    check_validation_limits()

    # Test payload info
    await test_document_upload_payload()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    issues = []

    if not agent_ok:
        issues.append("🔴 CRITICAL: Agent service is not reachable")

    if not rag_ok:
        issues.append(
            "🟡 WARNING: RAG service is not reachable (needed for document tests)"
        )

    if not env_ok:
        issues.append("🟡 WARNING: Some environment variables are missing")

    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Fix these issues before running the test suite")
        return False
    else:
        print("\n✅ All checks passed!")
        print("💡 You can now run: pytest tests/performance/test_load_testing.py -v")
        return True


if __name__ == "__main__":
    # Load environment variables from .env if it exists
    try:
        from dotenv import load_dotenv

        if Path(".env").exists():
            load_dotenv()
            print("📄 Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment only")

    success = asyncio.run(main())
    exit(0 if success else 1)
