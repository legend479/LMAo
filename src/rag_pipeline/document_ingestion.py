"""
Document Ingestion Service
Handles document discovery, batch processing, and ingestion pipeline
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

from .document_processor import (
    DocumentProcessor,
    ProcessedDocument,
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion"""

    source_directories: List[str]
    supported_extensions: List[str]
    max_file_size_mb: int = 100
    max_concurrent_files: int = 5
    skip_hidden_files: bool = True
    recursive_scan: bool = True
    metadata_file_patterns: List[str] = None
    exclude_patterns: List[str] = None

    def __post_init__(self):
        if self.metadata_file_patterns is None:
            self.metadata_file_patterns = ["*.json", "*.yaml", "*.yml"]
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "__pycache__",
                ".git",
                ".svn",
                "node_modules",
                ".DS_Store",
            ]


@dataclass
class IngestionResult:
    """Result of document ingestion process"""

    total_files_found: int
    total_files_processed: int
    successful_ingestions: int
    failed_ingestions: int
    skipped_files: int
    processing_time: float
    processed_documents: List[ProcessedDocument]
    errors: List[str]
    file_statistics: Dict[str, int]


class DocumentIngestionService:
    """Service for discovering and ingesting documents from various sources"""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.document_processor = DocumentProcessor()
        self._ingested_files: Set[str] = set()
        self._file_metadata_cache: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize the ingestion service"""
        logger.info("Initializing Document Ingestion Service")
        await self.document_processor.initialize()
        logger.info("Document Ingestion Service initialized")

    async def discover_documents(
        self, source_path: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Discover documents in a source directory with metadata"""

        discovered_files = []
        source_path = Path(source_path)

        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return discovered_files

        logger.info(f"Discovering documents in: {source_path}")

        # Load metadata files first
        metadata_map = await self._load_metadata_files(source_path)

        # Scan for documents
        if self.config.recursive_scan:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in source_path.glob(pattern):
            if await self._should_process_file(file_path):
                # Get metadata for this file
                relative_path = str(file_path.relative_to(source_path))
                file_metadata = metadata_map.get(relative_path, {})

                # Add file system metadata
                file_metadata.update(await self._extract_filesystem_metadata(file_path))

                discovered_files.append((str(file_path), file_metadata))

        logger.info(f"Discovered {len(discovered_files)} documents in {source_path}")
        return discovered_files

    async def ingest_documents(
        self, source_paths: Optional[List[str]] = None
    ) -> IngestionResult:
        """Ingest documents from configured or specified source paths"""

        start_time = datetime.utcnow()

        # Use provided paths or default to configured paths
        paths_to_process = source_paths or self.config.source_directories

        all_discovered_files = []

        # Discover files from all source paths
        for source_path in paths_to_process:
            discovered = await self.discover_documents(source_path)
            all_discovered_files.extend(discovered)

        logger.info(f"Total files discovered: {len(all_discovered_files)}")

        # Filter out already processed files
        files_to_process = []
        skipped_count = 0

        for file_path, metadata in all_discovered_files:
            if file_path not in self._ingested_files:
                files_to_process.append((file_path, metadata))
            else:
                skipped_count += 1

        logger.info(
            f"Files to process: {len(files_to_process)}, skipped: {skipped_count}"
        )

        # Process files in batches
        processed_documents = []
        all_errors = []
        successful_count = 0
        failed_count = 0

        if files_to_process:
            # Separate file paths and metadata
            file_paths = [fp for fp, _ in files_to_process]
            metadata_list = [md for _, md in files_to_process]

            # Process batch
            batch_result = await self.document_processor.process_batch(
                file_paths,
                metadata_list,
                max_concurrent=self.config.max_concurrent_files,
            )

            successful_count = batch_result.successful
            failed_count = batch_result.failed
            all_errors.extend(batch_result.errors)

            # Collect processed documents
            for result in batch_result.results:
                if result["status"] == "success":
                    # Mark as ingested
                    self._ingested_files.add(result["file_path"])

                    # Get the actual processed document (this is simplified - in real implementation
                    # you'd want to store these properly)
                    # For now, we'll create a summary
                    processed_documents.append(
                        {
                            "document_id": result["document_id"],
                            "file_path": result["file_path"],
                            "chunk_count": result["chunk_count"],
                            "processing_time": result["processing_time"],
                            "content_hash": result["content_hash"],
                        }
                    )

        # Calculate statistics
        file_stats = await self._calculate_file_statistics(all_discovered_files)
        total_time = (datetime.utcnow() - start_time).total_seconds()

        result = IngestionResult(
            total_files_found=len(all_discovered_files),
            total_files_processed=len(files_to_process),
            successful_ingestions=successful_count,
            failed_ingestions=failed_count,
            skipped_files=skipped_count,
            processing_time=total_time,
            processed_documents=processed_documents,
            errors=all_errors,
            file_statistics=file_stats,
        )

        logger.info(
            f"Ingestion completed: {successful_count} successful, {failed_count} failed, {total_time:.2f}s"
        )

        return result

    async def ingest_single_file(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Ingest a single document file"""

        logger.info(f"Ingesting single file: {file_path}")

        # Check if file should be processed
        if not await self._should_process_file(Path(file_path)):
            raise ValueError(
                f"File {file_path} should not be processed based on current configuration"
            )

        # Add filesystem metadata if not provided
        if metadata is None:
            metadata = {}

        metadata.update(await self._extract_filesystem_metadata(Path(file_path)))

        # Process the document
        processed_doc = await self.document_processor.process_document(
            file_path, metadata
        )

        # Mark as ingested
        self._ingested_files.add(file_path)

        logger.info(f"Successfully ingested: {file_path}")
        return processed_doc

    async def _should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed"""

        # Skip if not a file
        if not file_path.is_file():
            return False

        # Skip hidden files if configured
        if self.config.skip_hidden_files and file_path.name.startswith("."):
            return False

        # Check file extension
        if file_path.suffix.lower() not in self.config.supported_extensions:
            return False

        # Check file size
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                logger.warning(f"File too large: {file_path} ({file_size_mb:.1f}MB)")
                return False
        except OSError:
            logger.warning(f"Cannot access file: {file_path}")
            return False

        # Check exclude patterns
        file_str = str(file_path)
        for pattern in self.config.exclude_patterns:
            if pattern in file_str:
                return False

        return True

    async def _load_metadata_files(
        self, source_path: Path
    ) -> Dict[str, Dict[str, Any]]:
        """Load metadata files from the source directory"""

        metadata_map = {}

        for pattern in self.config.metadata_file_patterns:
            for metadata_file in source_path.glob(pattern):
                try:
                    if metadata_file.suffix.lower() == ".json":
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata_data = json.load(f)
                    else:
                        # For YAML files (would need PyYAML)
                        logger.warning(
                            f"YAML metadata files not yet supported: {metadata_file}"
                        )
                        continue

                    # Map metadata to files
                    if isinstance(metadata_data, dict):
                        for file_pattern, file_metadata in metadata_data.items():
                            # Simple pattern matching (could be improved with glob patterns)
                            metadata_map[file_pattern] = file_metadata

                except Exception as e:
                    logger.warning(f"Failed to load metadata file {metadata_file}: {e}")

        return metadata_map

    async def _extract_filesystem_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from filesystem"""

        try:
            stat = file_path.stat()
            return {
                "file_size": stat.st_size,
                "creation_time": datetime.fromtimestamp(stat.st_ctime),
                "modification_time": datetime.fromtimestamp(stat.st_mtime),
                "file_extension": file_path.suffix.lower(),
                "file_name": file_path.name,
                "directory": str(file_path.parent),
                "relative_path": str(file_path),
            }
        except OSError as e:
            logger.warning(
                f"Failed to extract filesystem metadata for {file_path}: {e}"
            )
            return {}

    async def _calculate_file_statistics(
        self, discovered_files: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, int]:
        """Calculate statistics about discovered files"""

        stats = {
            "total_files": len(discovered_files),
            "by_extension": {},
            "by_size_range": {
                "small_0_1mb": 0,
                "medium_1_10mb": 0,
                "large_10_100mb": 0,
                "xlarge_100mb_plus": 0,
            },
        }

        for file_path, metadata in discovered_files:
            # Count by extension
            ext = Path(file_path).suffix.lower()
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

            # Count by size range
            file_size_mb = metadata.get("file_size", 0) / (1024 * 1024)
            if file_size_mb < 1:
                stats["by_size_range"]["small_0_1mb"] += 1
            elif file_size_mb < 10:
                stats["by_size_range"]["medium_1_10mb"] += 1
            elif file_size_mb < 100:
                stats["by_size_range"]["large_10_100mb"] += 1
            else:
                stats["by_size_range"]["xlarge_100mb_plus"] += 1

        return stats

    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status"""

        processor_stats = await self.document_processor.get_processing_stats()

        return {
            "ingested_files_count": len(self._ingested_files),
            "processor_stats": processor_stats,
            "config": {
                "source_directories": self.config.source_directories,
                "supported_extensions": self.config.supported_extensions,
                "max_file_size_mb": self.config.max_file_size_mb,
                "max_concurrent_files": self.config.max_concurrent_files,
            },
        }

    async def reset_ingestion_state(self):
        """Reset ingestion state (for testing or re-ingestion)"""
        self._ingested_files.clear()
        self._file_metadata_cache.clear()
        await self.document_processor.clear_cache()
        logger.info("Ingestion state reset")

    async def shutdown(self):
        """Shutdown the ingestion service"""
        logger.info("Shutting down Document Ingestion Service")
        await self.document_processor.shutdown()
