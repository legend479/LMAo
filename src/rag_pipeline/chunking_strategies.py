"""
Advanced Chunking Strategies
Multi-granularity chunking with content-aware strategies and quality assessment
"""

import re
import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .models import Chunk, DocumentMetadata, ChunkingConfig
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""

    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    CONTENT_AWARE = "content_aware"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class ChunkingConfigAdvanced:
    """Advanced configuration for chunking strategies"""

    strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    chunk_sizes: Dict[str, int] = None
    overlap_ratios: Dict[str, float] = None
    min_chunk_size: int = 50
    max_chunk_size: int = 4096
    preserve_structure: bool = True
    quality_threshold: float = 0.7

    def __post_init__(self):
        if self.chunk_sizes is None:
            self.chunk_sizes = {"large": 2048, "medium": 512, "small": 128}
        if self.overlap_ratios is None:
            self.overlap_ratios = {
                "large": 0.1,  # 10% overlap
                "medium": 0.15,  # 15% overlap
                "small": 0.2,  # 20% overlap
            }


@dataclass
class ChunkQuality:
    """Quality metrics for a chunk"""

    coherence_score: float  # How well the chunk holds together semantically
    completeness_score: float  # How complete the information is
    boundary_score: float  # How well the boundaries are chosen
    readability_score: float  # How readable the chunk is
    overall_score: float  # Overall quality score

    @classmethod
    def calculate(cls, chunk: Chunk, context: str = "") -> "ChunkQuality":
        """Calculate quality metrics for a chunk"""
        content = chunk.content

        # Coherence: measure sentence connectivity and topic consistency
        coherence = cls._calculate_coherence(content)

        # Completeness: measure if chunk contains complete thoughts/structures
        completeness = cls._calculate_completeness(content, chunk.chunk_type)

        # Boundary: measure if chunk boundaries are at natural break points
        boundary = cls._calculate_boundary_quality(content, context)

        # Readability: measure how easy it is to understand
        readability = cls._calculate_readability(content)

        # Overall score (weighted average)
        overall = (
            coherence * 0.3 + completeness * 0.3 + boundary * 0.2 + readability * 0.2
        )

        return cls(
            coherence_score=coherence,
            completeness_score=completeness,
            boundary_score=boundary,
            readability_score=readability,
            overall_score=overall,
        )

    @staticmethod
    def _calculate_coherence(content: str) -> float:
        """Calculate coherence score based on sentence connectivity"""
        sentences = content.split(".")
        if len(sentences) < 2:
            return 1.0

        # Simple coherence: check for connecting words and consistent terminology
        connecting_words = [
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "additionally",
            "consequently",
            "thus",
            "hence",
        ]

        coherence_indicators = 0
        total_sentences = len(sentences) - 1

        for sentence in sentences[1:]:  # Skip first sentence
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in connecting_words):
                coherence_indicators += 1

        # Also check for repeated key terms (indicates topic consistency)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only consider longer words
                word_freq[word] = word_freq.get(word, 0) + 1

        repeated_terms = sum(1 for freq in word_freq.values() if freq > 1)
        term_consistency = min(repeated_terms / max(len(word_freq), 1), 1.0)

        connection_score = coherence_indicators / max(total_sentences, 1)
        return min((connection_score + term_consistency) / 2, 1.0)

    @staticmethod
    def _calculate_completeness(content: str, chunk_type: str) -> float:
        """Calculate completeness score based on content type"""
        if chunk_type == "code":
            # For code: check if functions/classes are complete
            try:
                # Try to parse as Python code
                ast.parse(content)
                return 1.0  # Valid Python code is considered complete
            except:
                # Check for common code patterns
                if any(
                    pattern in content for pattern in ["def ", "class ", "function"]
                ):
                    # Check if braces/brackets are balanced
                    open_chars = (
                        content.count("{") + content.count("(") + content.count("[")
                    )
                    close_chars = (
                        content.count("}") + content.count(")") + content.count("]")
                    )
                    balance_score = 1.0 - abs(open_chars - close_chars) / max(
                        open_chars + close_chars, 1
                    )
                    return max(balance_score, 0.3)
                return 0.7  # Assume reasonable completeness for other code

        elif chunk_type == "text":
            # For text: check if sentences are complete
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            if not sentences:
                return 0.0

            complete_sentences = 0
            for sentence in sentences:
                # A complete sentence should start with capital and have reasonable length
                if (
                    len(sentence) > 10
                    and sentence[0].isupper()
                    and sentence.endswith((".", "!", "?", ":", ";"))
                ):
                    complete_sentences += 1

            return complete_sentences / len(sentences)

        return 0.8  # Default completeness for other types

    @staticmethod
    def _calculate_boundary_quality(content: str, context: str) -> float:
        """Calculate how well chunk boundaries are chosen"""
        if not context:
            return 0.8  # Default if no context

        # Check if chunk starts and ends at natural boundaries
        start_score = 0.5
        end_score = 0.5

        # Good starting points
        content_start = content.strip()[:50].lower()
        if any(
            content_start.startswith(start)
            for start in ["# ", "## ", "### ", "def ", "class ", "function ", "import "]
        ):
            start_score = 1.0
        elif content_start[0].isupper():
            start_score = 0.8

        # Good ending points
        content_end = content.strip()[-50:].lower()
        if any(content_end.endswith(end) for end in [".", "!", "?", "}", ")", "]"]):
            end_score = 1.0
        elif content_end.endswith(":"):
            end_score = 0.6  # Colon might indicate incomplete thought

        return (start_score + end_score) / 2

    @staticmethod
    def _calculate_readability(content: str) -> float:
        """Calculate readability score (simplified Flesch-like metric)"""
        words = content.split()
        sentences = content.split(".")

        if not words or not sentences:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Count syllables (simplified: count vowel groups)
        syllables = 0
        for word in words:
            word_syllables = len(re.findall(r"[aeiouAEIOU]+", word))
            syllables += max(word_syllables, 1)  # At least 1 syllable per word

        avg_syllables_per_word = syllables / len(words)

        # Simplified Flesch formula (higher is more readable)
        flesch_score = (
            206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        )

        # Normalize to 0-1 scale (assuming 0-100 Flesch scale)
        return max(min(flesch_score / 100, 1.0), 0.0)


class BaseChunkingStrategy(ABC):
    """Base class for chunking strategies"""

    def __init__(self, config: ChunkingConfigAdvanced):
        self.config = config

    @abstractmethod
    async def create_chunks(
        self, content: str, doc_id: str, metadata: DocumentMetadata
    ) -> List[Chunk]:
        """Create chunks using this strategy"""
        pass

    def _validate_chunk_quality(self, chunk: Chunk, context: str = "") -> bool:
        """Validate if chunk meets quality threshold"""
        quality = ChunkQuality.calculate(chunk, context)
        return quality.overall_score >= self.config.quality_threshold

    def _create_chunk_metadata(
        self, base_metadata: DocumentMetadata, chunk_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create metadata for a chunk"""
        return {
            "document_title": base_metadata.title,
            "document_author": base_metadata.author,
            "document_category": base_metadata.category,
            "creation_date": (
                base_metadata.creation_date.isoformat()
                if base_metadata.creation_date
                else None
            ),
            "file_size": base_metadata.file_size,
            "mime_type": base_metadata.mime_type,
            **chunk_info,
        }


class HierarchicalChunkingStrategy(BaseChunkingStrategy):
    """Hierarchical chunking with parent-child relationships"""

    async def create_chunks(
        self, content: str, doc_id: str, metadata: DocumentMetadata
    ) -> List[Chunk]:
        """Create hierarchical chunks with quality validation"""

        chunks = []

        # Create large chunks first
        large_chunks = await self._create_size_chunks(
            content, doc_id, metadata, "large"
        )
        chunks.extend(large_chunks)

        # Create medium chunks as children of large chunks
        for large_chunk in large_chunks:
            medium_chunks = await self._create_size_chunks(
                large_chunk.content,
                doc_id,
                metadata,
                "medium",
                parent_chunk_id=large_chunk.chunk_id,
            )
            chunks.extend(medium_chunks)

            # Update parent with child references
            large_chunk.child_chunk_ids.extend(
                [chunk.chunk_id for chunk in medium_chunks]
            )

            # Create small chunks as children of medium chunks
            for medium_chunk in medium_chunks:
                small_chunks = await self._create_size_chunks(
                    medium_chunk.content,
                    doc_id,
                    metadata,
                    "small",
                    parent_chunk_id=medium_chunk.chunk_id,
                )
                chunks.extend(small_chunks)

                # Update parent with child references
                medium_chunk.child_chunk_ids.extend(
                    [chunk.chunk_id for chunk in small_chunks]
                )

        # Filter chunks by quality
        quality_chunks = []
        for chunk in chunks:
            if self._validate_chunk_quality(chunk, content):
                quality_chunks.append(chunk)
            else:
                logger.debug(f"Chunk {chunk.chunk_id} filtered out due to low quality")

        logger.info(
            f"Created {len(quality_chunks)} quality chunks from {len(chunks)} total chunks"
        )
        return quality_chunks

    async def _create_size_chunks(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        size_name: str,
        parent_chunk_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Create chunks of a specific size with intelligent boundaries"""

        chunk_size = self.config.chunk_sizes[size_name]
        overlap_ratio = self.config.overlap_ratios[size_name]
        overlap = int(chunk_size * overlap_ratio)

        chunks = []

        # Determine if content is code or text
        is_code = self._is_code_content(content)

        if is_code:
            chunks = await self._chunk_code_content(
                content,
                chunk_size,
                size_name,
                doc_id,
                metadata,
                parent_chunk_id,
                overlap,
            )
        else:
            chunks = await self._chunk_text_content(
                content,
                chunk_size,
                size_name,
                doc_id,
                metadata,
                parent_chunk_id,
                overlap,
            )

        return chunks

    def _is_code_content(self, content: str) -> bool:
        """Enhanced code detection"""
        code_indicators = [
            "def ",
            "class ",
            "function",
            "import ",
            "from ",
            "const ",
            "var ",
            "let ",
            "public ",
            "private ",
            "protected ",
            "static ",
            "void ",
            "int ",
            "string ",
            "package ",
            "#include",
            "using ",
            "namespace ",
            "struct ",
            "enum ",
            "{}",
            "[]",
            "()",
            ";",
            "//",
            "/*",
            "*/",
            "<!--",
            "-->",
        ]

        content_lower = content.lower()
        code_score = sum(
            1 for indicator in code_indicators if indicator in content_lower
        )

        # Also check for indentation patterns (common in code)
        lines = content.split("\n")
        indented_lines = sum(1 for line in lines if line.startswith(("    ", "\t")))
        indentation_ratio = indented_lines / max(len(lines), 1)

        # Consider it code if many indicators or high indentation
        return code_score > (len(content) / 1000) * 3 or indentation_ratio > 0.3

    async def _chunk_code_content(
        self,
        content: str,
        chunk_size: int,
        size_name: str,
        doc_id: str,
        metadata: DocumentMetadata,
        parent_chunk_id: Optional[str],
        overlap: int,
    ) -> List[Chunk]:
        """Enhanced code chunking with AST awareness"""

        chunks = []
        lines = content.split("\n")

        # Try to parse as Python for better structure awareness
        try:
            tree = ast.parse(content)
            chunks = await self._chunk_python_ast(
                content, tree, chunk_size, size_name, doc_id, metadata, parent_chunk_id
            )
        except:
            # Fall back to line-based chunking for other languages
            chunks = await self._chunk_code_lines(
                lines, chunk_size, size_name, doc_id, metadata, parent_chunk_id, overlap
            )

        return chunks

    async def _chunk_python_ast(
        self,
        content: str,
        tree: ast.AST,
        chunk_size: int,
        size_name: str,
        doc_id: str,
        metadata: DocumentMetadata,
        parent_chunk_id: Optional[str],
    ) -> List[Chunk]:
        """Chunk Python code using AST for better structure preservation"""

        chunks = []
        lines = content.split("\n")

        # Extract top-level definitions
        definitions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                start_line = node.lineno - 1  # AST is 1-indexed
                end_line = (
                    node.end_lineno if hasattr(node, "end_lineno") else start_line + 10
                )
                definitions.append(
                    (start_line, end_line, type(node).__name__, node.name)
                )

        # Sort by line number
        definitions.sort(key=lambda x: x[0])

        current_chunk_lines = []
        current_word_count = 0
        chunk_num = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            line_words = len(line.split())

            # Check if this line starts a definition that would exceed chunk size
            definition_at_line = next((d for d in definitions if d[0] == i), None)

            if definition_at_line and current_chunk_lines:
                # If we have content and a new definition would make chunk too large
                definition_size = definition_at_line[1] - definition_at_line[0]
                if (
                    current_word_count + definition_size * 5 > chunk_size
                ):  # Estimate words
                    # Create chunk with current content
                    chunk = await self._create_code_chunk(
                        current_chunk_lines,
                        chunk_num,
                        size_name,
                        doc_id,
                        metadata,
                        parent_chunk_id,
                        i - len(current_chunk_lines),
                        i,
                    )
                    chunks.append(chunk)
                    chunk_num += 1
                    current_chunk_lines = []
                    current_word_count = 0

            current_chunk_lines.append(line)
            current_word_count += line_words

            # Create chunk if it's getting too large
            if current_word_count >= chunk_size and current_chunk_lines:
                chunk = await self._create_code_chunk(
                    current_chunk_lines,
                    chunk_num,
                    size_name,
                    doc_id,
                    metadata,
                    parent_chunk_id,
                    i - len(current_chunk_lines) + 1,
                    i + 1,
                )
                chunks.append(chunk)
                chunk_num += 1
                current_chunk_lines = []
                current_word_count = 0

            i += 1

        # Handle remaining content
        if current_chunk_lines:
            chunk = await self._create_code_chunk(
                current_chunk_lines,
                chunk_num,
                size_name,
                doc_id,
                metadata,
                parent_chunk_id,
                len(lines) - len(current_chunk_lines),
                len(lines),
            )
            chunks.append(chunk)

        return chunks

    async def _chunk_code_lines(
        self,
        lines: List[str],
        chunk_size: int,
        size_name: str,
        doc_id: str,
        metadata: DocumentMetadata,
        parent_chunk_id: Optional[str],
        overlap: int,
    ) -> List[Chunk]:
        """Fallback line-based code chunking"""

        chunks = []
        current_chunk_lines = []
        current_word_count = 0
        chunk_num = 0

        for i, line in enumerate(lines):
            line_words = len(line.split())

            # Check for natural break points
            should_break = False
            if current_word_count > chunk_size * 0.8:  # 80% of target size
                # Look for function/class definitions or comments
                if (
                    line.strip().startswith(
                        ("def ", "class ", "function ", "//", "#", "/*")
                    )
                    or line.strip() == ""
                    or current_word_count >= chunk_size
                ):
                    should_break = True

            if should_break and current_chunk_lines:
                chunk = await self._create_code_chunk(
                    current_chunk_lines,
                    chunk_num,
                    size_name,
                    doc_id,
                    metadata,
                    parent_chunk_id,
                    i - len(current_chunk_lines),
                    i,
                )
                chunks.append(chunk)
                chunk_num += 1

                # Handle overlap
                if overlap > 0:
                    overlap_lines = current_chunk_lines[
                        -min(overlap // 10, len(current_chunk_lines)) :
                    ]
                    current_chunk_lines = overlap_lines
                    current_word_count = sum(
                        len(line.split()) for line in overlap_lines
                    )
                else:
                    current_chunk_lines = []
                    current_word_count = 0

            current_chunk_lines.append(line)
            current_word_count += line_words

        # Handle remaining content
        if current_chunk_lines:
            chunk = await self._create_code_chunk(
                current_chunk_lines,
                chunk_num,
                size_name,
                doc_id,
                metadata,
                parent_chunk_id,
                len(lines) - len(current_chunk_lines),
                len(lines),
            )
            chunks.append(chunk)

        return chunks

    async def _create_code_chunk(
        self,
        lines: List[str],
        chunk_num: int,
        size_name: str,
        doc_id: str,
        metadata: DocumentMetadata,
        parent_chunk_id: Optional[str],
        start_idx: int,
        end_idx: int,
    ) -> Chunk:
        """Create a code chunk with proper metadata"""

        content = "\n".join(lines)

        chunk_id = f"{doc_id}_chunk_{size_name}_{chunk_num}"
        if parent_chunk_id:
            chunk_id = f"{parent_chunk_id}_sub_{chunk_num}"

        # Analyze code content
        code_type = self._detect_code_type(content)

        chunk_metadata = self._create_chunk_metadata(
            metadata,
            {
                "size_category": size_name,
                "word_count": len(content.split()),
                "char_count": len(content),
                "line_count": len(lines),
                "code_type": code_type,
                "document_id": doc_id,
            },
        )

        return Chunk(
            content=content,
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=chunk_num,
            start_char=start_idx,
            end_char=end_idx,
            metadata=metadata,
            chunk_type="code",
            parent_chunk_id=parent_chunk_id,
        )

    def _detect_code_type(self, content: str) -> str:
        """Detect the type of code content"""
        content_lower = content.lower()

        if any(
            keyword in content_lower
            for keyword in ["def ", "import ", "class ", "from "]
        ):
            return "python"
        elif any(
            keyword in content_lower
            for keyword in ["function ", "const ", "let ", "var "]
        ):
            return "javascript"
        elif any(
            keyword in content_lower
            for keyword in ["public ", "private ", "class ", "void "]
        ):
            return "java"
        elif any(
            keyword in content_lower for keyword in ["#include", "int main", "printf"]
        ):
            return "c"
        elif any(
            keyword in content_lower for keyword in ["using ", "namespace ", "std::"]
        ):
            return "cpp"
        else:
            return "generic"

    async def _chunk_text_content(
        self,
        content: str,
        chunk_size: int,
        size_name: str,
        doc_id: str,
        metadata: DocumentMetadata,
        parent_chunk_id: Optional[str],
        overlap: int,
    ) -> List[Chunk]:
        """Enhanced text chunking with semantic awareness"""

        chunks = []

        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            # Fall back to sentence-based chunking
            return await self._chunk_by_sentences(
                content,
                chunk_size,
                size_name,
                doc_id,
                metadata,
                parent_chunk_id,
                overlap,
            )

        current_chunk_paragraphs = []
        current_word_count = 0
        chunk_num = 0

        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())

            # If single paragraph is too large, split it
            if paragraph_words > chunk_size:
                # Create chunk with current content if any
                if current_chunk_paragraphs:
                    chunk = await self._create_text_chunk(
                        "\n\n".join(current_chunk_paragraphs),
                        chunk_num,
                        size_name,
                        doc_id,
                        metadata,
                        parent_chunk_id,
                    )
                    chunks.append(chunk)
                    chunk_num += 1
                    current_chunk_paragraphs = []
                    current_word_count = 0

                # Split large paragraph by sentences
                large_para_chunks = await self._chunk_by_sentences(
                    paragraph,
                    chunk_size,
                    size_name,
                    doc_id,
                    metadata,
                    parent_chunk_id,
                    overlap,
                    chunk_num,
                )
                chunks.extend(large_para_chunks)
                chunk_num += len(large_para_chunks)
                continue

            # Check if adding this paragraph would exceed chunk size
            if (
                current_word_count + paragraph_words > chunk_size
                and current_chunk_paragraphs
            ):
                # Create chunk with current content
                chunk = await self._create_text_chunk(
                    "\n\n".join(current_chunk_paragraphs),
                    chunk_num,
                    size_name,
                    doc_id,
                    metadata,
                    parent_chunk_id,
                )
                chunks.append(chunk)
                chunk_num += 1

                # Handle overlap
                if overlap > 0 and current_chunk_paragraphs:
                    overlap_paras = current_chunk_paragraphs[
                        -1:
                    ]  # Keep last paragraph for overlap
                    current_chunk_paragraphs = overlap_paras
                    current_word_count = sum(len(p.split()) for p in overlap_paras)
                else:
                    current_chunk_paragraphs = []
                    current_word_count = 0

            current_chunk_paragraphs.append(paragraph)
            current_word_count += paragraph_words

        # Handle remaining content
        if current_chunk_paragraphs:
            chunk = await self._create_text_chunk(
                "\n\n".join(current_chunk_paragraphs),
                chunk_num,
                size_name,
                doc_id,
                metadata,
                parent_chunk_id,
            )
            chunks.append(chunk)

        return chunks

    async def _chunk_by_sentences(
        self,
        content: str,
        chunk_size: int,
        size_name: str,
        doc_id: str,
        metadata: DocumentMetadata,
        parent_chunk_id: Optional[str],
        overlap: int,
        start_chunk_num: int = 0,
    ) -> List[Chunk]:
        """Chunk content by sentences with overlap"""

        chunks = []
        sentences = self._split_into_sentences(content)

        current_chunk_sentences = []
        current_word_count = 0
        chunk_num = start_chunk_num

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If single sentence is too large, split it further
            if sentence_words > chunk_size:
                # Create chunk with current content if any
                if current_chunk_sentences:
                    chunk = await self._create_text_chunk(
                        " ".join(current_chunk_sentences),
                        chunk_num,
                        size_name,
                        doc_id,
                        metadata,
                        parent_chunk_id,
                    )
                    chunks.append(chunk)
                    chunk_num += 1
                    current_chunk_sentences = []
                    current_word_count = 0

                # Split large sentence by words
                words = sentence.split()
                for i in range(0, len(words), chunk_size):
                    word_chunk = " ".join(words[i : i + chunk_size])
                    chunk = await self._create_text_chunk(
                        word_chunk,
                        chunk_num,
                        size_name,
                        doc_id,
                        metadata,
                        parent_chunk_id,
                    )
                    chunks.append(chunk)
                    chunk_num += 1
                continue

            # Check if adding this sentence would exceed chunk size
            if (
                current_word_count + sentence_words > chunk_size
                and current_chunk_sentences
            ):
                # Create chunk with current content
                chunk = await self._create_text_chunk(
                    " ".join(current_chunk_sentences),
                    chunk_num,
                    size_name,
                    doc_id,
                    metadata,
                    parent_chunk_id,
                )
                chunks.append(chunk)
                chunk_num += 1

                # Handle overlap
                if overlap > 0 and current_chunk_sentences:
                    overlap_sentences = current_chunk_sentences[
                        -min(overlap // 20, len(current_chunk_sentences)) :
                    ]
                    current_chunk_sentences = overlap_sentences
                    current_word_count = sum(len(s.split()) for s in overlap_sentences)
                else:
                    current_chunk_sentences = []
                    current_word_count = 0

            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words

        # Handle remaining content
        if current_chunk_sentences:
            chunk = await self._create_text_chunk(
                " ".join(current_chunk_sentences),
                chunk_num,
                size_name,
                doc_id,
                metadata,
                parent_chunk_id,
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting with better boundary detection"""
        # Use regex for better sentence boundary detection
        sentence_endings = r"[.!?]+(?:\s+|$)"
        sentences = re.split(sentence_endings, text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    async def _create_text_chunk(
        self,
        content: str,
        chunk_num: int,
        size_name: str,
        doc_id: str,
        metadata: DocumentMetadata,
        parent_chunk_id: Optional[str],
    ) -> Chunk:
        """Create a text chunk with proper metadata"""

        chunk_id = f"{doc_id}_chunk_{size_name}_{chunk_num}"
        if parent_chunk_id:
            chunk_id = f"{parent_chunk_id}_sub_{chunk_num}"

        # Analyze text content
        text_type = self._detect_text_type(content)

        chunk_metadata = self._create_chunk_metadata(
            metadata,
            {
                "size_category": size_name,
                "word_count": len(content.split()),
                "char_count": len(content),
                "sentence_count": len(self._split_into_sentences(content)),
                "text_type": text_type,
                "document_id": doc_id,
            },
        )

        return Chunk(
            content=content,
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=chunk_num,
            start_char=0,  # Would need more context to calculate exact indices
            end_char=len(content),
            metadata=metadata,
            chunk_type="text",
            parent_chunk_id=parent_chunk_id,
        )

    def _detect_text_type(self, content: str) -> str:
        """Detect the type of text content"""
        content_lower = content.lower()

        if content.startswith("#"):
            return "markdown_header"
        elif any(marker in content for marker in ["- ", "* ", "1. ", "2. "]):
            return "list"
        elif "|" in content and content.count("|") > 3:
            return "table"
        elif any(
            keyword in content_lower
            for keyword in ["algorithm", "function", "method", "class"]
        ):
            return "technical"
        elif any(
            keyword in content_lower
            for keyword in ["example", "tutorial", "guide", "how to"]
        ):
            return "instructional"
        else:
            return "general"


class ChunkingManager:
    """Manager for different chunking strategies"""

    def __init__(self, config: ChunkingConfig = None):
        # Convert basic config to advanced config
        if config:
            advanced_config = ChunkingConfigAdvanced(
                strategy=ChunkingStrategy.HIERARCHICAL,
                min_chunk_size=config.min_chunk_size,
                max_chunk_size=config.max_chunk_size,
                preserve_structure=config.preserve_structure,
                quality_threshold=config.quality_threshold,
            )
        else:
            advanced_config = ChunkingConfigAdvanced()

        self.config = advanced_config
        self.strategies = {
            ChunkingStrategy.HIERARCHICAL: HierarchicalChunkingStrategy(self.config)
        }

    async def create_chunks(
        self,
        content: str,
        doc_id: str,
        metadata: DocumentMetadata,
        strategy: ChunkingStrategy = None,
    ) -> List[Chunk]:
        """Create chunks using specified strategy"""

        strategy = strategy or self.config.strategy

        if strategy not in self.strategies:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        chunking_strategy = self.strategies[strategy]
        chunks = await chunking_strategy.create_chunks(content, doc_id, metadata)

        # Validate and assess chunk quality
        quality_report = await self._assess_chunk_quality(chunks, content)

        logger.info(
            f"Created {len(chunks)} chunks using {strategy} strategy. "
            f"Average quality: {quality_report['average_quality']:.2f}"
        )

        return chunks

    async def _assess_chunk_quality(
        self, chunks: List[Chunk], original_content: str
    ) -> Dict[str, Any]:
        """Assess overall quality of chunking"""

        if not chunks:
            return {"average_quality": 0.0, "quality_distribution": {}}

        quality_scores = []
        quality_distribution = {"high": 0, "medium": 0, "low": 0}

        for chunk in chunks:
            quality = ChunkQuality.calculate(chunk, original_content)
            quality_scores.append(quality.overall_score)

            if quality.overall_score >= 0.8:
                quality_distribution["high"] += 1
            elif quality.overall_score >= 0.6:
                quality_distribution["medium"] += 1
            else:
                quality_distribution["low"] += 1

        return {
            "average_quality": sum(quality_scores) / len(quality_scores),
            "quality_distribution": quality_distribution,
            "total_chunks": len(chunks),
            "quality_scores": quality_scores,
        }
