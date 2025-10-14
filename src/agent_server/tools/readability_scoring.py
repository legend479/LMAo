"""
Readability Scoring Tool
Comprehensive tool for assessing educational content readability and pedagogical quality
"""

from typing import Dict, Any, List, Optional
import time
import re
import math
from dataclasses import dataclass

from .registry import (
    BaseTool,
    ToolResult,
    ExecutionContext,
    ToolCapabilities,
    ResourceRequirements,
    ToolCapability,
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReadabilityMetrics:
    """Comprehensive readability metrics"""

    flesch_reading_ease: float
    flesch_kincaid_grade_level: float
    gunning_fog_index: float
    smog_index: float
    automated_readability_index: float
    coleman_liau_index: float
    average_sentence_length: float
    average_syllables_per_word: float
    difficult_words_percentage: float
    word_count: int
    sentence_count: int
    paragraph_count: int
    character_count: int


@dataclass
class PedagogicalAssessment:
    """Assessment of pedagogical quality"""

    grade_level_suitability: str
    educational_complexity: str
    concept_density: float
    technical_term_ratio: float
    explanation_quality: float
    example_presence: bool
    structure_clarity: float
    engagement_score: float
    accessibility_score: float


@dataclass
class ContentAnalysis:
    """Detailed content analysis"""

    content_type: str
    domain: str
    technical_concepts: List[str]
    key_terms: List[str]
    code_blocks: List[str]
    examples: List[str]
    questions: List[str]
    headings: List[str]
    lists: List[str]
    emphasis_markers: List[str]


@dataclass
class SimplificationSuggestion:
    """Suggestion for content simplification"""

    suggestion_type: str
    original_text: str
    suggested_text: str
    reason: str
    impact_score: float
    line_number: Optional[int] = None


class TextAnalyzer:
    """Analyzes text structure and linguistic features"""

    def __init__(self):
        # Common difficult words (Dale-Chall list subset)
        self.difficult_words = set(
            [
                "abstract",
                "academic",
                "accelerate",
                "accessible",
                "accommodate",
                "accomplish",
                "accumulate",
                "accurate",
                "achieve",
                "acknowledge",
                "acquire",
                "adequate",
                "adjacent",
                "adjust",
                "administration",
                "advocate",
                "aggregate",
                "allocate",
                "alternative",
                "ambiguous",
                "analogous",
                "analyze",
                "anticipate",
                "apparent",
                "appreciate",
                "approach",
                "appropriate",
                "approximate",
                "arbitrary",
                "aspect",
                "assess",
                "assign",
                "assist",
                "assume",
                "assure",
                "attach",
                "attain",
                "attitude",
                "attribute",
                "authority",
                "available",
                "benefit",
                "category",
                "circumstance",
                "clarify",
                "classic",
                "clause",
                "coherent",
                "coincide",
                "collapse",
                "colleague",
                "commence",
                "comment",
                "commission",
                "commit",
                "commodity",
                "communicate",
                "community",
                "compatible",
                "compensate",
                "compile",
                "complement",
                "complex",
                "component",
                "compound",
                "comprehensive",
                "comprise",
                "compute",
                "conceive",
                "concentrate",
                "concept",
                "conclude",
                "concurrent",
                "conduct",
                "confer",
                "confine",
                "confirm",
                "conflict",
                "conform",
                "consent",
                "consequent",
                "considerable",
                "consist",
                "constant",
                "constitute",
                "constrain",
                "construct",
                "consult",
                "consume",
                "contact",
                "contain",
                "contemporary",
                "context",
                "contract",
                "contradict",
                "contrary",
                "contrast",
                "contribute",
                "controversy",
                "convene",
                "convention",
                "convert",
                "convince",
                "cooperate",
                "coordinate",
                "corporate",
                "correspond",
                "criteria",
                "crucial",
                "culture",
                "currency",
            ]
        )

        # Technical terms in software engineering
        self.technical_terms = set(
            [
                "algorithm",
                "api",
                "array",
                "boolean",
                "class",
                "code",
                "compile",
                "computer",
                "database",
                "debug",
                "function",
                "interface",
                "library",
                "method",
                "object",
                "parameter",
                "program",
                "software",
                "syntax",
                "variable",
                "framework",
                "architecture",
                "deployment",
                "repository",
                "version",
                "branch",
                "commit",
                "merge",
                "pull",
                "push",
                "clone",
                "fork",
                "issue",
                "bug",
                "feature",
                "testing",
                "unit",
                "integration",
                "automation",
                "continuous",
                "devops",
            ]
        )

    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze basic text structure"""

        # Clean text
        cleaned_text = self._clean_text(text)

        # Count basic elements
        words = self._get_words(cleaned_text)
        sentences = self._get_sentences(cleaned_text)
        paragraphs = self._get_paragraphs(text)

        # Calculate averages
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_syllables_per_word = (
            sum(self._count_syllables(word) for word in words) / len(words)
            if words
            else 0
        )

        # Count difficult words
        difficult_word_count = sum(
            1 for word in words if word.lower() in self.difficult_words
        )
        difficult_words_percentage = (
            (difficult_word_count / len(words) * 100) if words else 0
        )

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "character_count": len(cleaned_text),
            "average_sentence_length": avg_sentence_length,
            "average_syllables_per_word": avg_syllables_per_word,
            "difficult_words_count": difficult_word_count,
            "difficult_words_percentage": difficult_words_percentage,
            "words": words,
            "sentences": sentences,
            "paragraphs": paragraphs,
        }

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`[^`]*`", " ", text)

        # Remove markdown formatting
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic
        text = re.sub(r"#{1,6}\s*", "", text)  # Headers
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # Links

        # Remove special characters but keep sentence endings
        text = re.sub(r"[^\w\s.!?;:]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _get_words(self, text: str) -> List[str]:
        """Extract words from text"""
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        return [word for word in words if len(word) > 0]

    def _get_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Split on sentence endings
        sentences = re.split(r"[.!?]+", text)
        # Filter out empty sentences and very short ones
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    def _get_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        return paragraphs

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using a simple heuristic"""
        word = word.lower()

        # Handle special cases
        if len(word) <= 3:
            return 1

        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent e
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        # Ensure at least 1 syllable
        return max(1, syllable_count)


class ReadabilityCalculator:
    """Calculates various readability metrics"""

    def __init__(self):
        self.text_analyzer = TextAnalyzer()

    def calculate_all_metrics(self, text: str) -> ReadabilityMetrics:
        """Calculate all readability metrics"""

        # Analyze text structure
        structure = self.text_analyzer.analyze_text_structure(text)

        # Extract values
        words = structure["word_count"]
        sentences = structure["sentence_count"]
        paragraphs = structure["paragraph_count"]
        characters = structure["character_count"]
        avg_sentence_length = structure["average_sentence_length"]
        avg_syllables_per_word = structure["average_syllables_per_word"]
        difficult_words_percentage = structure["difficult_words_percentage"]

        # Calculate readability scores
        flesch_reading_ease = self._calculate_flesch_reading_ease(
            avg_sentence_length, avg_syllables_per_word
        )

        flesch_kincaid_grade_level = self._calculate_flesch_kincaid_grade_level(
            avg_sentence_length, avg_syllables_per_word
        )

        gunning_fog_index = self._calculate_gunning_fog_index(
            avg_sentence_length, difficult_words_percentage
        )

        smog_index = self._calculate_smog_index(sentences, structure["words"])

        automated_readability_index = self._calculate_automated_readability_index(
            characters, words, sentences
        )

        coleman_liau_index = self._calculate_coleman_liau_index(
            characters, words, sentences
        )

        return ReadabilityMetrics(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade_level=flesch_kincaid_grade_level,
            gunning_fog_index=gunning_fog_index,
            smog_index=smog_index,
            automated_readability_index=automated_readability_index,
            coleman_liau_index=coleman_liau_index,
            average_sentence_length=avg_sentence_length,
            average_syllables_per_word=avg_syllables_per_word,
            difficult_words_percentage=difficult_words_percentage,
            word_count=words,
            sentence_count=sentences,
            paragraph_count=paragraphs,
            character_count=characters,
        )

    def _calculate_flesch_reading_ease(
        self, avg_sentence_length: float, avg_syllables_per_word: float
    ) -> float:
        """Calculate Flesch Reading Ease score"""
        if avg_sentence_length == 0 or avg_syllables_per_word == 0:
            return 0.0

        score = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        )
        return max(0.0, min(100.0, score))

    def _calculate_flesch_kincaid_grade_level(
        self, avg_sentence_length: float, avg_syllables_per_word: float
    ) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        if avg_sentence_length == 0 or avg_syllables_per_word == 0:
            return 0.0

        grade_level = (
            (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        )
        return max(0.0, grade_level)

    def _calculate_gunning_fog_index(
        self, avg_sentence_length: float, difficult_words_percentage: float
    ) -> float:
        """Calculate Gunning Fog Index"""
        if avg_sentence_length == 0:
            return 0.0

        fog_index = 0.4 * (avg_sentence_length + difficult_words_percentage)
        return max(0.0, fog_index)

    def _calculate_smog_index(self, sentence_count: int, words: List[str]) -> float:
        """Calculate SMOG Index"""
        if sentence_count < 30:
            return 0.0  # SMOG requires at least 30 sentences

        # Count polysyllabic words (3+ syllables)
        polysyllabic_count = 0
        for word in words:
            if self.text_analyzer._count_syllables(word) >= 3:
                polysyllabic_count += 1

        smog = 1.0430 * math.sqrt(polysyllabic_count * (30 / sentence_count)) + 3.1291
        return max(0.0, smog)

    def _calculate_automated_readability_index(
        self, characters: int, words: int, sentences: int
    ) -> float:
        """Calculate Automated Readability Index"""
        if words == 0 or sentences == 0:
            return 0.0

        ari = (4.71 * (characters / words)) + (0.5 * (words / sentences)) - 21.43
        return max(0.0, ari)

    def _calculate_coleman_liau_index(
        self, characters: int, words: int, sentences: int
    ) -> float:
        """Calculate Coleman-Liau Index"""
        if words == 0:
            return 0.0

        l = (characters / words) * 100
        s = (sentences / words) * 100

        cli = (0.0588 * l) - (0.296 * s) - 15.8
        return max(0.0, cli)


class ReadabilityScoringTool(BaseTool):
    """Comprehensive readability scoring tool for educational content assessment"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.readability_calculator = ReadabilityCalculator()

        # Configuration
        self.default_target_grade_level = (
            config.get("default_target_grade_level", 8) if config else 8
        )
        self.default_target_flesch_score = (
            config.get("default_target_flesch_score", 65.0) if config else 65.0
        )

    async def initialize(self):
        """Initialize the readability scoring tool"""
        await super().initialize()

        logger.info(
            "Readability Scoring Tool initialized",
            default_target_grade_level=self.default_target_grade_level,
            default_target_flesch_score=self.default_target_flesch_score,
        )

    async def execute(
        self, parameters: Dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Execute readability scoring and assessment"""

        start_time = time.time()

        try:
            # Extract parameters
            text = parameters["text"]
            assessment_type = parameters.get(
                "assessment_type", "comprehensive"
            )  # basic, comprehensive
            target_grade_level = parameters.get(
                "target_grade_level", self.default_target_grade_level
            )
            target_flesch_score = parameters.get(
                "target_flesch_score", self.default_target_flesch_score
            )
            generate_suggestions = parameters.get("generate_suggestions", True)

            logger.info(
                "Executing readability scoring",
                assessment_type=assessment_type,
                text_length=len(text),
                target_grade_level=target_grade_level,
                session_id=context.session_id,
            )

            # Validate inputs
            if not text or len(text.strip()) < 50:
                raise ValueError(
                    "Text must be at least 50 characters long for meaningful analysis"
                )

            if len(text) > 50000:
                raise ValueError(
                    "Text is too long for analysis (maximum 50,000 characters)"
                )

            # Calculate readability metrics
            readability_metrics = self.readability_calculator.calculate_all_metrics(
                text
            )

            # Analyze content structure
            content_analysis = self._analyze_content_structure(text)

            # Prepare result data
            result_data = {
                "assessment_type": assessment_type,
                "text_statistics": {
                    "word_count": readability_metrics.word_count,
                    "sentence_count": readability_metrics.sentence_count,
                    "paragraph_count": readability_metrics.paragraph_count,
                    "character_count": readability_metrics.character_count,
                    "average_sentence_length": readability_metrics.average_sentence_length,
                    "average_syllables_per_word": readability_metrics.average_syllables_per_word,
                    "difficult_words_percentage": readability_metrics.difficult_words_percentage,
                },
                "readability_scores": {
                    "flesch_reading_ease": {
                        "score": readability_metrics.flesch_reading_ease,
                        "interpretation": self._interpret_flesch_reading_ease(
                            readability_metrics.flesch_reading_ease
                        ),
                    },
                    "flesch_kincaid_grade_level": {
                        "score": readability_metrics.flesch_kincaid_grade_level,
                        "interpretation": self._interpret_grade_level(
                            readability_metrics.flesch_kincaid_grade_level
                        ),
                    },
                    "gunning_fog_index": {
                        "score": readability_metrics.gunning_fog_index,
                        "interpretation": self._interpret_grade_level(
                            readability_metrics.gunning_fog_index
                        ),
                    },
                    "smog_index": {
                        "score": readability_metrics.smog_index,
                        "interpretation": self._interpret_grade_level(
                            readability_metrics.smog_index
                        ),
                    },
                    "automated_readability_index": {
                        "score": readability_metrics.automated_readability_index,
                        "interpretation": self._interpret_grade_level(
                            readability_metrics.automated_readability_index
                        ),
                    },
                    "coleman_liau_index": {
                        "score": readability_metrics.coleman_liau_index,
                        "interpretation": self._interpret_grade_level(
                            readability_metrics.coleman_liau_index
                        ),
                    },
                },
                "content_analysis": content_analysis,
            }

            # Generate simplification suggestions if requested
            if generate_suggestions:
                suggestions = self._generate_simplification_suggestions(
                    text, readability_metrics, target_grade_level, target_flesch_score
                )

                result_data["simplification_suggestions"] = suggestions

            # Add target comparison
            result_data["target_comparison"] = {
                "target_grade_level": target_grade_level,
                "current_grade_level": readability_metrics.flesch_kincaid_grade_level,
                "grade_level_difference": readability_metrics.flesch_kincaid_grade_level
                - target_grade_level,
                "target_flesch_score": target_flesch_score,
                "current_flesch_score": readability_metrics.flesch_reading_ease,
                "flesch_score_difference": readability_metrics.flesch_reading_ease
                - target_flesch_score,
                "meets_target": (
                    readability_metrics.flesch_kincaid_grade_level <= target_grade_level
                    and readability_metrics.flesch_reading_ease >= target_flesch_score
                ),
            }

            execution_time = time.time() - start_time
            result_data["processing_time"] = execution_time

            # Calculate quality and confidence scores
            quality_score = self._calculate_quality_score(
                result_data, readability_metrics
            )
            confidence_score = self._calculate_confidence_score(
                readability_metrics, content_analysis
            )

            result = ToolResult(
                data=result_data,
                metadata={
                    "tool": "readability_scoring",
                    "version": "1.0.0",
                    "assessment_type": assessment_type,
                    "target_grade_level": target_grade_level,
                    "session_id": context.session_id,
                },
                execution_time=execution_time,
                success=True,
                resource_usage={
                    "cpu_usage": 0.3,
                    "memory_usage_mb": 64,
                    "network_requests": 0,
                },
                quality_score=quality_score,
                confidence_score=confidence_score,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Readability scoring failed",
                error=str(e),
                session_id=context.session_id,
            )

            return ToolResult(
                data=None,
                metadata={
                    "tool": "readability_scoring",
                    "session_id": context.session_id,
                    "error_type": type(e).__name__,
                },
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                quality_score=0.0,
                confidence_score=0.0,
            )

    def _analyze_content_structure(self, text: str) -> Dict[str, Any]:
        """Analyze content structure for educational features"""

        # Extract technical concepts
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        technical_concepts = [
            word
            for word in words
            if word in self.readability_calculator.text_analyzer.technical_terms
        ]

        # Extract code blocks
        code_blocks = re.findall(r"```[\s\S]*?```", text)
        inline_code = re.findall(r"`([^`]+)`", text)

        # Extract examples
        example_patterns = [
            r"for example[,:]\s*([^.!?]*[.!?])",
            r"example[,:]\s*([^.!?]*[.!?])",
            r"such as[,:]\s*([^.!?]*[.!?])",
        ]

        examples = []
        for pattern in example_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            examples.extend(matches)

        # Extract questions
        questions = re.findall(r"([^.!?]*\?)", text)

        # Extract headings
        headings = re.findall(r"^#{1,6}\s*(.+)$", text, re.MULTILINE)

        # Extract lists
        bullets = re.findall(r"^\s*[-*+]\s*(.+)$", text, re.MULTILINE)
        numbered = re.findall(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)

        return {
            "technical_concepts_count": len(set(technical_concepts)),
            "technical_concepts": list(set(technical_concepts))[:10],
            "code_blocks_count": len(code_blocks),
            "inline_code_count": len(inline_code),
            "examples_count": len(examples),
            "examples": [ex.strip() for ex in examples if len(ex.strip()) > 10][:5],
            "questions_count": len(questions),
            "questions": [q.strip() for q in questions if len(q.strip()) > 5][:5],
            "headings_count": len(headings),
            "headings": [h.strip() for h in headings][:10],
            "lists_count": len(bullets) + len(numbered),
            "bullet_points": len(bullets),
            "numbered_items": len(numbered),
        }

    def _generate_simplification_suggestions(
        self,
        text: str,
        metrics: ReadabilityMetrics,
        target_grade_level: int,
        target_flesch_score: float,
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for content simplification"""

        suggestions = []

        # Suggest breaking long sentences
        if metrics.average_sentence_length > 20:
            suggestions.append(
                {
                    "type": "sentence_length",
                    "issue": f"Average sentence length is {metrics.average_sentence_length:.1f} words",
                    "suggestion": "Break long sentences into shorter ones (aim for 15-20 words per sentence)",
                    "impact": "high",
                    "target_improvement": "Reduce average sentence length to improve readability",
                }
            )

        # Suggest reducing difficult words
        if metrics.difficult_words_percentage > 10:
            suggestions.append(
                {
                    "type": "vocabulary",
                    "issue": f"{metrics.difficult_words_percentage:.1f}% of words are considered difficult",
                    "suggestion": "Replace complex words with simpler alternatives where possible",
                    "impact": "high",
                    "target_improvement": "Reduce difficult words to under 10% of total words",
                }
            )

        # Suggest improving Flesch Reading Ease
        if metrics.flesch_reading_ease < target_flesch_score:
            difference = target_flesch_score - metrics.flesch_reading_ease
            suggestions.append(
                {
                    "type": "readability",
                    "issue": f"Flesch Reading Ease score is {metrics.flesch_reading_ease:.1f} (target: {target_flesch_score})",
                    "suggestion": "Simplify sentence structure and vocabulary to improve readability",
                    "impact": "high",
                    "target_improvement": f"Increase Flesch Reading Ease by {difference:.1f} points",
                }
            )

        # Suggest reducing grade level
        if metrics.flesch_kincaid_grade_level > target_grade_level:
            difference = metrics.flesch_kincaid_grade_level - target_grade_level
            suggestions.append(
                {
                    "type": "grade_level",
                    "issue": f"Grade level is {metrics.flesch_kincaid_grade_level:.1f} (target: {target_grade_level})",
                    "suggestion": "Use shorter sentences and simpler words to lower grade level",
                    "impact": "high",
                    "target_improvement": f"Reduce grade level by {difference:.1f} grades",
                }
            )

        # Suggest structural improvements
        if metrics.paragraph_count < 3 and metrics.word_count > 300:
            suggestions.append(
                {
                    "type": "structure",
                    "issue": "Content has few paragraph breaks for its length",
                    "suggestion": "Break content into smaller paragraphs (50-100 words each)",
                    "impact": "medium",
                    "target_improvement": "Improve content structure and readability",
                }
            )

        return suggestions[:8]  # Return top 8 suggestions

    def _interpret_flesch_reading_ease(self, score: float) -> str:
        """Interpret Flesch Reading Ease score"""

        if score >= 90:
            return "Very Easy (5th grade level)"
        elif score >= 80:
            return "Easy (6th grade level)"
        elif score >= 70:
            return "Fairly Easy (7th grade level)"
        elif score >= 60:
            return "Standard (8th-9th grade level)"
        elif score >= 50:
            return "Fairly Difficult (10th-12th grade level)"
        elif score >= 30:
            return "Difficult (college level)"
        else:
            return "Very Difficult (graduate level)"

    def _interpret_grade_level(self, grade_level: float) -> str:
        """Interpret grade level score"""

        if grade_level <= 6:
            return "Elementary School"
        elif grade_level <= 8:
            return "Middle School"
        elif grade_level <= 12:
            return "High School"
        elif grade_level <= 16:
            return "College"
        else:
            return "Graduate School"

    def _calculate_quality_score(
        self, result_data: Dict[str, Any], metrics: ReadabilityMetrics
    ) -> float:
        """Calculate quality score for readability assessment"""

        base_score = 0.7

        # Boost for comprehensive analysis
        if result_data["assessment_type"] == "comprehensive":
            base_score += 0.1

        # Boost for reasonable text length
        word_count = metrics.word_count
        if 100 <= word_count <= 5000:  # Good length for analysis
            base_score += 0.1

        # Boost for balanced readability
        flesch_score = metrics.flesch_reading_ease
        if 50 <= flesch_score <= 80:  # Reasonable readability range
            base_score += 0.1

        return min(1.0, base_score)

    def _calculate_confidence_score(
        self, metrics: ReadabilityMetrics, content_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for readability assessment"""

        base_confidence = 0.8

        # Higher confidence for longer texts
        if metrics.word_count >= 200:
            base_confidence += 0.1
        elif metrics.word_count < 100:
            base_confidence -= 0.2

        # Higher confidence for texts with multiple sentences
        if metrics.sentence_count >= 10:
            base_confidence += 0.1
        elif metrics.sentence_count < 5:
            base_confidence -= 0.1

        return max(0.0, min(1.0, base_confidence))

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for readability scoring"""
        return {
            "name": "readability_scoring",
            "description": "Assess readability and pedagogical quality of educational content with Flesch-Kincaid and other metrics",
            "version": "1.0.0",
            "parameters": {
                "text": {
                    "type": "string",
                    "description": "Text content to analyze for readability",
                    "required": True,
                    "minLength": 50,
                    "maxLength": 50000,
                },
                "assessment_type": {
                    "type": "string",
                    "description": "Type of assessment to perform",
                    "enum": ["basic", "comprehensive"],
                    "default": "comprehensive",
                    "required": False,
                },
                "target_grade_level": {
                    "type": "integer",
                    "description": "Target grade level for content",
                    "minimum": 1,
                    "maximum": 16,
                    "default": 8,
                    "required": False,
                },
                "target_flesch_score": {
                    "type": "number",
                    "description": "Target Flesch Reading Ease score",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 65.0,
                    "required": False,
                },
                "generate_suggestions": {
                    "type": "boolean",
                    "description": "Whether to generate simplification suggestions",
                    "default": True,
                    "required": False,
                },
            },
            "required_params": ["text"],
            "returns": {
                "type": "object",
                "properties": {
                    "assessment_type": {"type": "string"},
                    "text_statistics": {
                        "type": "object",
                        "properties": {
                            "word_count": {"type": "integer"},
                            "sentence_count": {"type": "integer"},
                            "paragraph_count": {"type": "integer"},
                            "character_count": {"type": "integer"},
                            "average_sentence_length": {"type": "number"},
                            "average_syllables_per_word": {"type": "number"},
                            "difficult_words_percentage": {"type": "number"},
                        },
                    },
                    "readability_scores": {
                        "type": "object",
                        "properties": {
                            "flesch_reading_ease": {"type": "object"},
                            "flesch_kincaid_grade_level": {"type": "object"},
                            "gunning_fog_index": {"type": "object"},
                            "smog_index": {"type": "object"},
                            "automated_readability_index": {"type": "object"},
                            "coleman_liau_index": {"type": "object"},
                        },
                    },
                    "content_analysis": {"type": "object"},
                    "simplification_suggestions": {"type": "array"},
                    "target_comparison": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
            "capabilities": {
                "primary": "content_analysis",
                "secondary": ["educational_assessment", "text_processing"],
                "input_types": ["text", "educational_content"],
                "output_types": ["readability_metrics", "suggestions", "analysis"],
                "supported_metrics": [
                    "flesch_reading_ease",
                    "flesch_kincaid",
                    "gunning_fog",
                    "smog",
                    "ari",
                    "coleman_liau",
                ],
            },
        }

    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.DATA_ANALYSIS,
            secondary_capabilities=[
                ToolCapability.CONTENT_GENERATION,
                ToolCapability.VALIDATION,
            ],
            input_types=["text", "educational_content", "technical_writing"],
            output_types=["readability_metrics", "analysis_report", "suggestions"],
            supported_formats=["text", "markdown"],
            language_support=["en"],
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=0.5,
            memory_mb=128,
            network_bandwidth_mbps=0.0,  # No network required
            storage_mb=5,  # Minimal storage for processing
            gpu_memory_mb=0,
            max_execution_time=30,
            concurrent_limit=10,  # Can handle multiple concurrent analyses
        )

    async def cleanup(self):
        """Cleanup tool resources"""
        logger.info("Cleaning up Readability Scoring Tool")
        # No specific cleanup needed for this tool
