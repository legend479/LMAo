"""
CLI tool for testing document ingestion pipeline
"""

import asyncio
import click
import json
from typing import List

from .document_ingestion import DocumentIngestionService, IngestionConfig
from ..shared.logging import get_logger

logger = get_logger(__name__)


@click.group()
def cli():
    """Document Processing CLI"""
    pass


@cli.command()
@click.argument("source_paths", nargs=-1, required=True)
@click.option(
    "--extensions",
    "-e",
    multiple=True,
    default=[".pdf", ".docx", ".pptx", ".txt", ".md"],
    help="File extensions to process",
)
@click.option("--max-size", default=100, help="Maximum file size in MB")
@click.option("--max-concurrent", default=5, help="Maximum concurrent files to process")
@click.option("--output", "-o", help="Output file for results (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
async def ingest(
    source_paths: List[str],
    extensions: List[str],
    max_size: int,
    max_concurrent: int,
    output: str,
    verbose: bool,
):
    """Ingest documents from source paths"""

    # Configure logging level
    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = IngestionConfig(
        source_directories=list(source_paths),
        supported_extensions=list(extensions),
        max_file_size_mb=max_size,
        max_concurrent_files=max_concurrent,
    )

    # Initialize service
    service = DocumentIngestionService(config)
    await service.initialize()

    try:
        # Run ingestion
        click.echo(f"Starting ingestion from: {', '.join(source_paths)}")
        result = await service.ingest_documents()

        # Display results
        click.echo("\n=== Ingestion Results ===")
        click.echo(f"Total files found: {result.total_files_found}")
        click.echo(f"Files processed: {result.total_files_processed}")
        click.echo(f"Successful: {result.successful_ingestions}")
        click.echo(f"Failed: {result.failed_ingestions}")
        click.echo(f"Skipped: {result.skipped_files}")
        click.echo(f"Processing time: {result.processing_time:.2f}s")

        # Show file statistics
        click.echo("\n=== File Statistics ===")
        for ext, count in result.file_statistics.get("by_extension", {}).items():
            click.echo(f"{ext}: {count} files")

        # Show errors if any
        if result.errors:
            click.echo("\n=== Errors ===")
            for error in result.errors[:10]:  # Show first 10 errors
                click.echo(f"- {error}")
            if len(result.errors) > 10:
                click.echo(f"... and {len(result.errors) - 10} more errors")

        # Save results to file if requested
        if output:
            output_data = {
                "summary": {
                    "total_files_found": result.total_files_found,
                    "total_files_processed": result.total_files_processed,
                    "successful_ingestions": result.successful_ingestions,
                    "failed_ingestions": result.failed_ingestions,
                    "skipped_files": result.skipped_files,
                    "processing_time": result.processing_time,
                },
                "file_statistics": result.file_statistics,
                "processed_documents": result.processed_documents,
                "errors": result.errors,
            }

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            click.echo(f"\nResults saved to: {output}")

    finally:
        await service.shutdown()


@cli.command()
@click.argument("file_path")
@click.option("--metadata", "-m", help="JSON metadata file")
@click.option("--output", "-o", help="Output file for results (JSON)")
async def process_file(file_path: str, metadata: str, output: str):
    """Process a single document file"""

    # Load metadata if provided
    file_metadata = {}
    if metadata:
        with open(metadata, "r") as f:
            file_metadata = json.load(f)

    # Create minimal configuration
    config = IngestionConfig(
        source_directories=[],
        supported_extensions=[".pdf", ".docx", ".pptx", ".txt", ".md"],
    )

    # Initialize service
    service = DocumentIngestionService(config)
    await service.initialize()

    try:
        # Process single file
        click.echo(f"Processing file: {file_path}")
        result = await service.ingest_single_file(file_path, file_metadata)

        # Display results
        click.echo("\n=== Processing Results ===")
        click.echo(f"Document ID: {result.document_id}")
        click.echo(f"Content length: {len(result.content)} characters")
        click.echo(f"Chunks created: {len(result.chunks)}")
        click.echo(f"Processing time: {result.processing_time:.2f}s")
        click.echo(f"Content hash: {result.content_hash}")

        if result.processing_errors:
            click.echo("\n=== Errors ===")
            for error in result.processing_errors:
                click.echo(f"- {error}")

        # Show chunk information
        click.echo("\n=== Chunk Information ===")
        chunk_types = {}
        for chunk in result.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

        for chunk_type, count in chunk_types.items():
            click.echo(f"{chunk_type}: {count} chunks")

        # Save results to file if requested
        if output:
            output_data = {
                "document_id": result.document_id,
                "original_path": result.original_path,
                "content_length": len(result.content),
                "chunk_count": len(result.chunks),
                "processing_time": result.processing_time,
                "content_hash": result.content_hash,
                "metadata": {
                    "title": result.metadata.title,
                    "author": result.metadata.author,
                    "word_count": result.metadata.word_count,
                    "file_size": result.metadata.file_size,
                    "mime_type": result.metadata.mime_type,
                },
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content_preview": (
                            chunk.content[:200] + "..."
                            if len(chunk.content) > 200
                            else chunk.content
                        ),
                        "chunk_type": chunk.chunk_type,
                        "word_count": chunk.metadata.get("word_count", 0),
                        "parent_chunk_id": chunk.parent_chunk_id,
                    }
                    for chunk in result.chunks
                ],
                "processing_errors": result.processing_errors,
            }

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            click.echo(f"\nResults saved to: {output}")

    finally:
        await service.shutdown()


@cli.command()
@click.argument("directory")
async def discover(directory: str):
    """Discover documents in a directory without processing"""

    config = IngestionConfig(
        source_directories=[directory],
        supported_extensions=[".pdf", ".docx", ".pptx", ".txt", ".md"],
    )

    service = DocumentIngestionService(config)
    await service.initialize()

    try:
        # Discover files
        discovered = await service.discover_documents(directory)

        click.echo(f"Discovered {len(discovered)} files in {directory}:")

        for file_path, metadata in discovered:
            file_size_mb = metadata.get("file_size", 0) / (1024 * 1024)
            click.echo(f"  {file_path} ({file_size_mb:.1f}MB)")

    finally:
        await service.shutdown()


def main():
    """Main entry point for async CLI"""
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the CLI with asyncio
    def run_async_cli():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cli(_anyio_backend="asyncio")
        finally:
            loop.close()

    run_async_cli()


if __name__ == "__main__":
    main()
