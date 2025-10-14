#!/usr/bin/env python3
"""
Test script for document ingestion pipeline
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_pipeline.document_ingestion import (
    DocumentIngestionService,
    IngestionConfig,
)
from src.rag_pipeline.main import RAGPipeline


async def test_single_file_processing():
    """Test processing a single file"""
    print("=== Testing Single File Processing ===")

    config = IngestionConfig(
        source_directories=[],
        supported_extensions=[".txt", ".py", ".md"],
        max_file_size_mb=10,
        max_concurrent_files=2,
    )

    service = DocumentIngestionService(config)
    await service.initialize()

    try:
        # Test text file
        print("\nProcessing sample.txt...")
        result = await service.ingest_single_file("test_documents/sample.txt")

        print(f"Document ID: {result.document_id}")
        print(f"Content length: {len(result.content)} characters")
        print(f"Chunks created: {len(result.chunks)}")
        print(f"Processing time: {result.processing_time:.2f}s")

        # Show chunk types
        chunk_types = {}
        for chunk in result.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

        print("Chunk types:", chunk_types)

        # Test Python file
        print("\nProcessing code_example.py...")
        result2 = await service.ingest_single_file("test_documents/code_example.py")

        print(f"Document ID: {result2.document_id}")
        print(f"Content length: {len(result2.content)} characters")
        print(f"Chunks created: {len(result2.chunks)}")
        print(f"Processing time: {result2.processing_time:.2f}s")

        # Show chunk types for code
        chunk_types2 = {}
        for chunk in result2.chunks:
            chunk_types2[chunk.chunk_type] = chunk_types2.get(chunk.chunk_type, 0) + 1

        print("Chunk types:", chunk_types2)

        # Show a sample chunk
        if result2.chunks:
            sample_chunk = result2.chunks[0]
            print("\nSample chunk preview:")
            print(f"Chunk ID: {sample_chunk.chunk_id}")
            print(f"Type: {sample_chunk.chunk_type}")
            print(f"Content preview: {sample_chunk.content[:200]}...")

    finally:
        await service.shutdown()


async def test_batch_processing():
    """Test batch processing"""
    print("\n=== Testing Batch Processing ===")

    config = IngestionConfig(
        source_directories=["test_documents"],
        supported_extensions=[".txt", ".py", ".md"],
        max_file_size_mb=10,
        max_concurrent_files=2,
    )

    service = DocumentIngestionService(config)
    await service.initialize()

    try:
        # Test directory ingestion
        print("\nIngesting from test_documents directory...")
        result = await service.ingest_documents()

        print(f"Total files found: {result.total_files_found}")
        print(f"Files processed: {result.total_files_processed}")
        print(f"Successful: {result.successful_ingestions}")
        print(f"Failed: {result.failed_ingestions}")
        print(f"Skipped: {result.skipped_files}")
        print(f"Processing time: {result.processing_time:.2f}s")

        print("\nFile statistics:")
        for ext, count in result.file_statistics.get("by_extension", {}).items():
            print(f"  {ext}: {count} files")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")

        # Show processed documents
        print(f"\nProcessed documents ({len(result.processed_documents)}):")
        for doc in result.processed_documents:
            print(f"  - {doc['document_id']}: {doc['chunk_count']} chunks")

    finally:
        await service.shutdown()


async def test_rag_pipeline():
    """Test the main RAG pipeline"""
    print("\n=== Testing RAG Pipeline ===")

    config = IngestionConfig(
        source_directories=["test_documents"],
        supported_extensions=[".txt", ".py", ".md"],
        max_file_size_mb=10,
        max_concurrent_files=2,
    )

    pipeline = RAGPipeline(config)
    await pipeline.initialize()

    try:
        # Test single document ingestion
        print("\nIngesting single document via pipeline...")
        result = await pipeline.ingest_document("test_documents/sample.txt")

        print(f"Status: {result['status']}")
        if result["status"] == "success":
            print(f"Document ID: {result['document_id']}")
            print(f"Chunks processed: {result['chunks_processed']}")
            print(f"Processing time: {result['processing_time']:.2f}s")

        # Test directory ingestion
        print("\nIngesting from directory via pipeline...")
        dir_result = await pipeline.ingest_from_directories(["test_documents"])

        print(f"Total files found: {dir_result['total_files_found']}")
        print(f"Successful ingestions: {dir_result['successful_ingestions']}")
        print(f"Failed ingestions: {dir_result['failed_ingestions']}")

        # Test health check
        print("\nChecking pipeline health...")
        health = await pipeline.health_check()
        print(f"Pipeline status: {health['pipeline']}")

        # Test stats
        print("\nGetting document stats...")
        stats = await pipeline.get_document_stats()
        print(
            f"Ingestion stats: {stats['ingestion_stats']['ingested_files_count']} files ingested"
        )

    finally:
        await pipeline.shutdown()


async def main():
    """Run all tests"""
    print("Starting Document Ingestion Pipeline Tests")
    print("=" * 50)

    try:
        await test_single_file_processing()
        await test_batch_processing()
        await test_rag_pipeline()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
