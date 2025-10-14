"""
Adaptive Content Generation System
Implements audience detection and content complexity adaptation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import asyncio
import uuid

from .prompt_engineering import (
    AdvancedPromptEngineeringFramework,
    AudienceLevel,
    ContentType,
    PromptGenerationRequest,
    PromptType,
)
from .tools.readability_scoring import ReadabilityScorer
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ComplexityLevel(Enum):
    """Content complexity levels"""

    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ContentDomain(Enum):
    """Content domain categories"""

    SOFTWARE_ENGINEERING = "software_engineering"
    PROGRAMMING = "programming"
    ALGORITHMS = "algorithms"
    DATA_STRUCTURES = "data_structures"
    SYSTEM_DESIGN = "system_design"
    WEB_DEVELOPMENT = "web_development"
    MOBILE_DEVELOPMENT = "mobile_development"
    DATABASE = "database"
    DEVOPS = "devops"
    SECURITY = "security"
    TESTING = "testing"
    GENERAL_CS = "general_cs"


class TransformationType(Enum):
    """Types of content transformations"""

    SIMPLIFY = "simplify"
    ELABORATE = "elaborate"
    ADJUST_TONE = "adjust_tone"
    ADD_EXAMPLES = "add_examples"
    REMOVE_JARGON = "remove_jargon"
    INCREASE_TECHNICAL_DEPTH = "increase_technical_depth"
    MAKE_CONVERSATIONAL = "make_conversational"
    MAKE_FORMAL = "make_formal"


@dataclass
class AudienceProfile:
    """Profile of the target audience"""

    level: AudienceLevel
    domain_knowledge: Dict[ContentDomain, float]  # 0.0 to 1.0
    preferred_complexity: ComplexityLevel
    learning_style: str  # visual, auditory, kinesthetic, reading
    technical_background: bool
    age_group: Optional[str] = None
    professional_context: Optional[str] = None
    specific_interests: List[str] = field(default_factory=list)


@dataclass
class ContentAnalysis:
    """Analysis of content characteristics"""

    complexity_score: float  # 0.0 to 1.0
    technical_density: float  # 0.0 to 1.0
    readability_score: float  # 0.0 to 1.0
    domain: ContentDomain
    key_concepts: List[str]
    jargon_terms: List[str]
    example_count: int
    explanation_depth: float  # 0.0 to 1.0
    tone: str  # formal, informal, conversational, academic
    length: int  # word count


@dataclass
class ContentGenerationRequest:
    """Request for adaptive content generation"""

    topic: str
    target_audience: AudienceProfile
    content_type: ContentType
    desired_length: Optional[int] = None
    include_examples: bool = True
    include_code: bool = False
    tone_preference: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedContent:
    """Result of content generation"""

    content_id: str
    content: str
    metadata: Dict[str, Any]
    analysis: ContentAnalysis
    generation_time: float
    confidence_score: float
    transformations_applied: List[TransformationType]
    readability_metrics: Dict[str, float]


class AudienceDetector:
    """Detects audience characteristics from context and requests"""

    def __init__(self):
        self.domain_keywords = self._initialize_domain_keywords()
        self.complexity_indicators = self._initialize_complexity_indicators()
        self.audience_patterns = self._initialize_audience_patterns()

    async def detect_audience(self, context: Dict[str, Any]) -> AudienceProfile:
        """Detect audience characteristics from context"""

        # Extract text content for analysis
        text_content = self._extract_text_content(context)

        # Detect audience level
        audience_level = await self._detect_audience_level(text_content, context)

        # Detect domain knowledge
        domain_knowledge = await self._detect_domain_knowledge(text_content, context)

        # Detect preferred complexity
        preferred_complexity = await self._detect_preferred_complexity(
            text_content, context
        )

        # Detect learning style
        learning_style = await self._detect_learning_style(context)

        # Detect technical background
        technical_background = await self._detect_technical_background(
            text_content, context
        )

        return AudienceProfile(
            level=audience_level,
            domain_knowledge=domain_knowledge,
            preferred_complexity=preferred_complexity,
            learning_style=learning_style,
            technical_background=technical_background,
            age_group=context.get("age_group"),
            professional_context=context.get("professional_context"),
            specific_interests=context.get("interests", []),
        )

    def _extract_text_content(self, context: Dict[str, Any]) -> str:
        """Extract text content from context for analysis"""

        text_parts = []

        # Extract from various context fields
        if "user_query" in context:
            text_parts.append(context["user_query"])

        if "conversation_history" in context:
            for message in context["conversation_history"][-5:]:  # Last 5 messages
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])

        if "user_profile" in context:
            profile = context["user_profile"]
            if isinstance(profile, dict):
                text_parts.extend(
                    [
                        profile.get("bio", ""),
                        profile.get("interests", ""),
                        profile.get("background", ""),
                    ]
                )

        return " ".join(text_parts)

    async def _detect_audience_level(
        self, text_content: str, context: Dict[str, Any]
    ) -> AudienceLevel:
        """Detect audience level from text and context"""

        # Check for explicit indicators
        if "student" in text_content.lower():
            if any(
                grade in text_content.lower() for grade in ["elementary", "primary"]
            ):
                return AudienceLevel.K12_ELEMENTARY
            elif any(grade in text_content.lower() for grade in ["middle", "junior"]):
                return AudienceLevel.K12_MIDDLE
            elif any(grade in text_content.lower() for grade in ["high", "senior"]):
                return AudienceLevel.K12_HIGH
            elif "undergraduate" in text_content.lower():
                return AudienceLevel.UNDERGRADUATE
            elif "graduate" in text_content.lower():
                return AudienceLevel.GRADUATE

        # Check for professional indicators
        if any(
            term in text_content.lower()
            for term in ["professional", "work", "job", "career"]
        ):
            return AudienceLevel.PROFESSIONAL

        # Analyze vocabulary complexity
        complex_words = len(re.findall(r"\b\w{8,}\b", text_content))
        total_words = len(text_content.split())

        if total_words > 0:
            complexity_ratio = complex_words / total_words

            if complexity_ratio > 0.3:
                return AudienceLevel.EXPERT
            elif complexity_ratio > 0.2:
                return AudienceLevel.ADVANCED
            elif complexity_ratio > 0.1:
                return AudienceLevel.INTERMEDIATE
            else:
                return AudienceLevel.BEGINNER

        return AudienceLevel.INTERMEDIATE  # Default

    async def _detect_domain_knowledge(
        self, text_content: str, context: Dict[str, Any]
    ) -> Dict[ContentDomain, float]:
        """Detect domain knowledge levels"""

        domain_knowledge = {}
        text_lower = text_content.lower()

        for domain, keywords in self.domain_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            knowledge_score = min(keyword_count / len(keywords), 1.0)
            domain_knowledge[domain] = knowledge_score

        return domain_knowledge

    async def _detect_preferred_complexity(
        self, text_content: str, context: Dict[str, Any]
    ) -> ComplexityLevel:
        """Detect preferred complexity level"""

        # Check for explicit preferences
        if any(term in text_content.lower() for term in ["simple", "basic", "easy"]):
            return ComplexityLevel.SIMPLE
        elif any(
            term in text_content.lower()
            for term in ["detailed", "comprehensive", "thorough"]
        ):
            return ComplexityLevel.COMPLEX
        elif any(
            term in text_content.lower() for term in ["advanced", "expert", "deep"]
        ):
            return ComplexityLevel.VERY_COMPLEX

        # Analyze question complexity
        question_words = len(
            re.findall(r"\b(what|how|why|when|where|which)\b", text_content.lower())
        )
        if question_words > 3:
            return ComplexityLevel.COMPLEX
        elif question_words > 1:
            return ComplexityLevel.MODERATE

        return ComplexityLevel.MODERATE  # Default

    async def _detect_learning_style(self, context: Dict[str, Any]) -> str:
        """Detect preferred learning style"""

        # Check for explicit preferences in context
        if "learning_style" in context:
            return context["learning_style"]

        # Infer from request patterns
        if context.get("include_diagrams") or context.get("visual_aids"):
            return "visual"
        elif context.get("include_audio") or context.get("pronunciation"):
            return "auditory"
        elif context.get("hands_on") or context.get("interactive"):
            return "kinesthetic"

        return "reading"  # Default

    async def _detect_technical_background(
        self, text_content: str, context: Dict[str, Any]
    ) -> bool:
        """Detect if audience has technical background"""

        technical_terms = [
            "algorithm",
            "api",
            "database",
            "framework",
            "library",
            "repository",
            "deployment",
            "architecture",
            "scalability",
            "optimization",
            "debugging",
            "version control",
            "continuous integration",
            "microservices",
        ]

        technical_count = sum(
            1 for term in technical_terms if term in text_content.lower()
        )
        return technical_count >= 2

    def _initialize_domain_keywords(self) -> Dict[ContentDomain, List[str]]:
        """Initialize domain-specific keywords"""

        return {
            ContentDomain.SOFTWARE_ENGINEERING: [
                "software",
                "engineering",
                "development",
                "lifecycle",
                "requirements",
                "design",
                "testing",
                "maintenance",
                "quality",
                "process",
            ],
            ContentDomain.PROGRAMMING: [
                "programming",
                "coding",
                "syntax",
                "variables",
                "functions",
                "loops",
                "conditions",
                "objects",
                "classes",
                "methods",
            ],
            ContentDomain.ALGORITHMS: [
                "algorithm",
                "complexity",
                "sorting",
                "searching",
                "recursion",
                "dynamic programming",
                "greedy",
                "divide and conquer",
            ],
            ContentDomain.DATA_STRUCTURES: [
                "array",
                "list",
                "stack",
                "queue",
                "tree",
                "graph",
                "hash table",
                "linked list",
                "heap",
                "trie",
            ],
            ContentDomain.SYSTEM_DESIGN: [
                "system design",
                "architecture",
                "scalability",
                "load balancing",
                "caching",
                "database design",
                "microservices",
                "distributed systems",
            ],
            ContentDomain.WEB_DEVELOPMENT: [
                "html",
                "css",
                "javascript",
                "react",
                "angular",
                "vue",
                "node.js",
                "frontend",
                "backend",
                "full stack",
            ],
            ContentDomain.DATABASE: [
                "database",
                "sql",
                "nosql",
                "mongodb",
                "postgresql",
                "mysql",
                "queries",
                "indexing",
                "normalization",
                "transactions",
            ],
            ContentDomain.DEVOPS: [
                "devops",
                "ci/cd",
                "docker",
                "kubernetes",
                "jenkins",
                "deployment",
                "infrastructure",
                "monitoring",
                "automation",
            ],
        }

    def _initialize_complexity_indicators(self) -> Dict[str, float]:
        """Initialize complexity indicators"""

        return {
            "simple": 0.2,
            "basic": 0.2,
            "easy": 0.2,
            "beginner": 0.3,
            "intermediate": 0.5,
            "advanced": 0.7,
            "expert": 0.9,
            "complex": 0.8,
            "sophisticated": 0.8,
            "comprehensive": 0.7,
        }

    def _initialize_audience_patterns(self) -> Dict[str, AudienceLevel]:
        """Initialize audience detection patterns"""

        return {
            "student": AudienceLevel.UNDERGRADUATE,
            "beginner": AudienceLevel.BEGINNER,
            "professional": AudienceLevel.PROFESSIONAL,
            "expert": AudienceLevel.EXPERT,
            "developer": AudienceLevel.INTERMEDIATE,
            "engineer": AudienceLevel.ADVANCED,
        }


class ContentAnalyzer:
    """Analyzes content characteristics and complexity"""

    def __init__(self):
        self.readability_scorer = ReadabilityScorer()
        self.technical_terms = self._load_technical_terms()
        self.jargon_patterns = self._load_jargon_patterns()

    async def analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content characteristics"""

        # Basic metrics
        word_count = len(content.split())

        # Complexity analysis
        complexity_score = await self._calculate_complexity_score(content)

        # Technical density
        technical_density = await self._calculate_technical_density(content)

        # Readability analysis
        readability_score = await self._calculate_readability_score(content)

        # Domain classification
        domain = await self._classify_domain(content)

        # Extract key concepts
        key_concepts = await self._extract_key_concepts(content)

        # Identify jargon terms
        jargon_terms = await self._identify_jargon_terms(content)

        # Count examples
        example_count = await self._count_examples(content)

        # Assess explanation depth
        explanation_depth = await self._assess_explanation_depth(content)

        # Detect tone
        tone = await self._detect_tone(content)

        return ContentAnalysis(
            complexity_score=complexity_score,
            technical_density=technical_density,
            readability_score=readability_score,
            domain=domain,
            key_concepts=key_concepts,
            jargon_terms=jargon_terms,
            example_count=example_count,
            explanation_depth=explanation_depth,
            tone=tone,
            length=word_count,
        )

    async def _calculate_complexity_score(self, content: str) -> float:
        """Calculate content complexity score"""

        factors = []

        # Sentence length
        sentences = re.split(r"[.!?]+", content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(
            len(sentences), 1
        )
        sentence_complexity = min(avg_sentence_length / 20, 1.0)
        factors.append(sentence_complexity)

        # Word complexity
        words = content.split()
        complex_words = [w for w in words if len(w) > 6]
        word_complexity = len(complex_words) / max(len(words), 1)
        factors.append(word_complexity)

        # Technical term density
        technical_count = sum(
            1 for term in self.technical_terms if term in content.lower()
        )
        technical_complexity = min(technical_count / max(len(words), 1) * 100, 1.0)
        factors.append(technical_complexity)

        # Nested structure complexity
        nested_indicators = [
            "however",
            "furthermore",
            "moreover",
            "nevertheless",
            "consequently",
        ]
        nested_count = sum(
            1 for indicator in nested_indicators if indicator in content.lower()
        )
        nested_complexity = min(nested_count / max(len(sentences), 1), 1.0)
        factors.append(nested_complexity)

        return sum(factors) / len(factors)

    async def _calculate_technical_density(self, content: str) -> float:
        """Calculate technical term density"""

        words = content.lower().split()
        technical_count = sum(1 for word in words if word in self.technical_terms)

        return technical_count / max(len(words), 1)

    async def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score using multiple metrics"""

        try:
            # Use the readability scorer tool
            flesch_score = await self.readability_scorer.calculate_flesch_reading_ease(
                content
            )

            # Convert Flesch score to 0-1 scale (higher = more readable)
            # Flesch scores: 90-100 (very easy), 0-30 (very difficult)
            normalized_score = flesch_score / 100

            return max(0.0, min(1.0, normalized_score))

        except Exception as e:
            logger.warning("Failed to calculate readability score", error=str(e))
            return 0.5  # Default moderate readability

    async def _classify_domain(self, content: str) -> ContentDomain:
        """Classify content domain"""

        content_lower = content.lower()
        domain_scores = {}

        # Domain keywords from AudienceDetector
        detector = AudienceDetector()
        domain_keywords = detector.domain_keywords

        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain

        return ContentDomain.GENERAL_CS  # Default

    async def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""

        # Simple extraction based on capitalized terms and technical terms
        concepts = []

        # Find capitalized terms (potential concepts)
        capitalized_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
        concepts.extend(capitalized_terms[:10])  # Limit to top 10

        # Find technical terms
        content_lower = content.lower()
        technical_concepts = [
            term for term in self.technical_terms if term in content_lower
        ]
        concepts.extend(technical_concepts[:10])  # Limit to top 10

        # Remove duplicates and return
        return list(set(concepts))

    async def _identify_jargon_terms(self, content: str) -> List[str]:
        """Identify jargon terms in content"""

        jargon_terms = []
        content_lower = content.lower()

        for pattern in self.jargon_patterns:
            matches = re.findall(pattern, content_lower)
            jargon_terms.extend(matches)

        return list(set(jargon_terms))

    async def _count_examples(self, content: str) -> int:
        """Count examples in content"""

        example_indicators = [
            r"for example",
            r"for instance",
            r"such as",
            r"like this:",
            r"here\'s an example",
            r"consider this",
            r"let\'s say",
        ]

        count = 0
        content_lower = content.lower()

        for pattern in example_indicators:
            count += len(re.findall(pattern, content_lower))

        return count

    async def _assess_explanation_depth(self, content: str) -> float:
        """Assess the depth of explanation"""

        depth_indicators = [
            "because",
            "since",
            "therefore",
            "thus",
            "consequently",
            "this means",
            "in other words",
            "specifically",
            "namely",
            "furthermore",
            "moreover",
            "additionally",
            "however",
        ]

        content_lower = content.lower()
        indicator_count = sum(
            1 for indicator in depth_indicators if indicator in content_lower
        )

        # Normalize by content length
        words = len(content.split())
        depth_score = (indicator_count / max(words, 1)) * 100

        return min(depth_score, 1.0)

    async def _detect_tone(self, content: str) -> str:
        """Detect the tone of content"""

        formal_indicators = [
            "therefore",
            "furthermore",
            "consequently",
            "thus",
            "hence",
        ]
        informal_indicators = [
            "you",
            "we",
            "let's",
            "here's",
            "that's",
            "don't",
            "can't",
        ]
        conversational_indicators = ["well", "now", "so", "okay", "right", "you know"]

        content_lower = content.lower()

        formal_count = sum(
            1 for indicator in formal_indicators if indicator in content_lower
        )
        informal_count = sum(
            1 for indicator in informal_indicators if indicator in content_lower
        )
        conversational_count = sum(
            1 for indicator in conversational_indicators if indicator in content_lower
        )

        if conversational_count > max(formal_count, informal_count):
            return "conversational"
        elif formal_count > informal_count:
            return "formal"
        elif informal_count > 0:
            return "informal"
        else:
            return "neutral"

    def _load_technical_terms(self) -> List[str]:
        """Load technical terms for analysis"""

        return [
            "algorithm",
            "api",
            "array",
            "backend",
            "cache",
            "class",
            "compiler",
            "database",
            "debugging",
            "deployment",
            "framework",
            "frontend",
            "function",
            "git",
            "hash",
            "http",
            "inheritance",
            "json",
            "kubernetes",
            "library",
            "method",
            "microservice",
            "object",
            "polymorphism",
            "query",
            "recursion",
            "repository",
            "server",
            "testing",
            "variable",
            "xml",
            "yaml",
        ]

    def _load_jargon_patterns(self) -> List[str]:
        """Load jargon patterns for identification"""

        return [
            r"\b\w+(?:ing|tion|sion|ness|ment|ity|ism)\b",  # Complex suffixes
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b\w*(?:ize|ise|ify)\b",  # Technical verbs
        ]


class ContentTransformer:
    """Transforms content for different audience levels and requirements"""

    def __init__(self):
        self.transformation_strategies = self._initialize_transformation_strategies()
        self.readability_scorer = ReadabilityScorer()

    async def transform_content(
        self,
        content: str,
        current_analysis: ContentAnalysis,
        target_audience: AudienceProfile,
        transformations: List[TransformationType],
    ) -> str:
        """Transform content based on target audience and requirements"""

        transformed_content = content

        for transformation in transformations:
            if transformation in self.transformation_strategies:
                strategy = self.transformation_strategies[transformation]
                transformed_content = await strategy(
                    transformed_content, current_analysis, target_audience
                )

        return transformed_content

    async def simplify_content(
        self,
        content: str,
        current_analysis: ContentAnalysis,
        target_audience: AudienceProfile,
    ) -> str:
        """Simplify content for lower complexity audiences"""

        # Replace jargon terms with simpler alternatives
        simplified = await self._replace_jargon(content, target_audience)

        # Break down complex sentences
        simplified = await self._simplify_sentences(simplified)

        # Add explanations for technical terms
        simplified = await self._add_term_explanations(simplified, target_audience)

        # Add more examples if needed
        if current_analysis.example_count < 2:
            simplified = await self._add_examples(simplified, target_audience)

        return simplified

    async def elaborate_content(
        self,
        content: str,
        current_analysis: ContentAnalysis,
        target_audience: AudienceProfile,
    ) -> str:
        """Elaborate content for higher complexity audiences"""

        # Add technical depth
        elaborated = await self._add_technical_depth(content, target_audience)

        # Include advanced concepts
        elaborated = await self._include_advanced_concepts(elaborated, target_audience)

        # Add detailed explanations
        elaborated = await self._add_detailed_explanations(elaborated)

        return elaborated

    async def adjust_tone(
        self,
        content: str,
        current_analysis: ContentAnalysis,
        target_audience: AudienceProfile,
    ) -> str:
        """Adjust content tone based on audience preferences"""

        target_tone = self._determine_target_tone(target_audience)

        if target_tone == "conversational":
            return await self._make_conversational(content)
        elif target_tone == "formal":
            return await self._make_formal(content)
        elif target_tone == "academic":
            return await self._make_academic(content)

        return content

    async def optimize_readability(
        self, content: str, target_score: float, max_iterations: int = 3
    ) -> Tuple[str, float]:
        """Iteratively optimize content readability"""

        current_content = content
        current_score = await self._get_readability_score(current_content)

        for iteration in range(max_iterations):
            if abs(current_score - target_score) < 0.1:
                break

            if current_score < target_score:
                # Need to make more readable (simpler)
                current_content = await self._improve_readability(current_content)
            else:
                # Content is already more readable than target
                break

            current_score = await self._get_readability_score(current_content)

        return current_content, current_score

    async def _replace_jargon(
        self, content: str, target_audience: AudienceProfile
    ) -> str:
        """Replace jargon terms with simpler alternatives"""

        jargon_replacements = {
            "utilize": "use",
            "implement": (
                "create"
                if target_audience.level
                in [AudienceLevel.BEGINNER, AudienceLevel.K12_ELEMENTARY]
                else "build"
            ),
            "instantiate": "create",
            "initialize": "set up",
            "optimize": "improve",
            "refactor": "reorganize",
            "deprecated": "outdated",
            "legacy": "old",
            "scalable": "able to grow",
            "robust": "strong and reliable",
        }

        result = content
        for jargon, replacement in jargon_replacements.items():
            result = re.sub(
                r"\b" + jargon + r"\b", replacement, result, flags=re.IGNORECASE
            )

        return result

    async def _simplify_sentences(self, content: str) -> str:
        """Break down complex sentences into simpler ones"""

        sentences = re.split(r"[.!?]+", content)
        simplified_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Split long sentences at conjunctions
            if len(sentence.split()) > 20:
                # Split at common conjunctions
                conjunctions = [", and ", ", but ", ", however ", ", therefore "]
                for conj in conjunctions:
                    if conj in sentence:
                        parts = sentence.split(conj, 1)
                        if len(parts) == 2:
                            simplified_sentences.append(parts[0].strip() + ".")
                            simplified_sentences.append(parts[1].strip())
                            break
                else:
                    simplified_sentences.append(sentence)
            else:
                simplified_sentences.append(sentence)

        return " ".join(simplified_sentences)

    async def _add_term_explanations(
        self, content: str, target_audience: AudienceProfile
    ) -> str:
        """Add explanations for technical terms"""

        technical_terms = {
            "API": "Application Programming Interface - a way for different software programs to communicate",
            "algorithm": "a step-by-step procedure for solving a problem",
            "database": "a structured collection of data stored electronically",
            "framework": "a pre-built structure that provides a foundation for building applications",
            "repository": "a storage location for code and project files",
            "deployment": "the process of making software available for use",
            "debugging": "finding and fixing errors in code",
        }

        # Only add explanations for beginner audiences
        if target_audience.level not in [
            AudienceLevel.BEGINNER,
            AudienceLevel.K12_ELEMENTARY,
            AudienceLevel.K12_MIDDLE,
        ]:
            return content

        result = content
        for term, explanation in technical_terms.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, result, re.IGNORECASE):
                replacement = f"{term} ({explanation})"
                result = re.sub(
                    pattern, replacement, result, count=1, flags=re.IGNORECASE
                )

        return result

    async def _add_examples(
        self, content: str, target_audience: AudienceProfile
    ) -> str:
        """Add relevant examples to content"""

        # Simple example addition based on content type
        if "function" in content.lower() and target_audience.technical_background:
            example = "\n\nFor example:\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n```"
            content += example
        elif "algorithm" in content.lower():
            example = "\n\nFor example, think of a recipe - it's a step-by-step algorithm for cooking a dish."
            content += example

        return content

    async def _add_technical_depth(
        self, content: str, target_audience: AudienceProfile
    ) -> str:
        """Add technical depth for advanced audiences"""

        if target_audience.level in [AudienceLevel.EXPERT, AudienceLevel.ADVANCED]:
            # Add implementation details, performance considerations, etc.
            depth_additions = {
                "algorithm": " Consider the time complexity (Big O notation) and space complexity when choosing algorithms.",
                "database": " Consider indexing strategies, normalization, and query optimization for better performance.",
                "API": " Consider rate limiting, authentication, versioning, and error handling in API design.",
            }

            result = content
            for term, addition in depth_additions.items():
                if term in content.lower():
                    result += addition

            return result

        return content

    async def _include_advanced_concepts(
        self, content: str, target_audience: AudienceProfile
    ) -> str:
        """Include advanced concepts for expert audiences"""

        if target_audience.level == AudienceLevel.EXPERT:
            # Add references to advanced topics
            advanced_concepts = {
                "design patterns": " Consider applying appropriate design patterns like Singleton, Factory, or Observer.",
                "scalability": " Consider horizontal vs vertical scaling, load balancing, and distributed systems architecture.",
                "security": " Implement proper authentication, authorization, input validation, and encryption.",
            }

            result = content
            for concept, addition in advanced_concepts.items():
                if any(word in content.lower() for word in concept.split()):
                    result += addition

            return result

        return content

    async def _add_detailed_explanations(self, content: str) -> str:
        """Add detailed explanations and reasoning"""

        # Add "why" explanations
        explanatory_additions = {
            "use": " This is important because it provides better maintainability and code reuse.",
            "avoid": " This helps prevent common errors and security vulnerabilities.",
            "implement": " This approach ensures better performance and scalability.",
        }

        result = content
        for trigger, addition in explanatory_additions.items():
            if trigger in content.lower():
                result += addition
                break  # Add only one explanation to avoid redundancy

        return result

    def _determine_target_tone(self, target_audience: AudienceProfile) -> str:
        """Determine appropriate tone for target audience"""

        if target_audience.level in [
            AudienceLevel.K12_ELEMENTARY,
            AudienceLevel.K12_MIDDLE,
        ]:
            return "conversational"
        elif target_audience.level in [AudienceLevel.GRADUATE, AudienceLevel.EXPERT]:
            return "academic"
        elif target_audience.level == AudienceLevel.PROFESSIONAL:
            return "formal"
        else:
            return "neutral"

    async def _make_conversational(self, content: str) -> str:
        """Make content more conversational"""

        # Add conversational elements
        conversational_replacements = {
            r"\bYou should\b": "You can",
            r"\bIt is important to\b": "It's helpful to",
            r"\bOne must\b": "You need to",
            r"\bThis will\b": "This'll",
            r"\bCannot\b": "Can't",
            r"\bDo not\b": "Don't",
        }

        result = content
        for pattern, replacement in conversational_replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Add conversational connectors
        if not any(word in result.lower() for word in ["let's", "we", "you"]):
            result = "Let's explore " + result.lower()

        return result

    async def _make_formal(self, content: str) -> str:
        """Make content more formal"""

        formal_replacements = {
            r"\bcan't\b": "cannot",
            r"\bdon't\b": "do not",
            r"\bwon't\b": "will not",
            r"\bit's\b": "it is",
            r"\bthat's\b": "that is",
            r"\blet's\b": "let us",
        }

        result = content
        for pattern, replacement in formal_replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    async def _make_academic(self, content: str) -> str:
        """Make content more academic"""

        # Add academic language patterns
        academic_additions = {
            "shows": "demonstrates",
            "uses": "utilizes",
            "helps": "facilitates",
            "makes": "enables",
        }

        result = content
        for informal, formal in academic_additions.items():
            result = re.sub(
                r"\b" + informal + r"\b", formal, result, flags=re.IGNORECASE
            )

        return result

    async def _improve_readability(self, content: str) -> str:
        """Improve content readability"""

        # Simplify sentences and vocabulary
        improved = await self._simplify_sentences(content)
        improved = await self._replace_jargon(
            improved,
            AudienceProfile(
                level=AudienceLevel.BEGINNER,
                domain_knowledge={},
                preferred_complexity=ComplexityLevel.SIMPLE,
                learning_style="reading",
                technical_background=False,
            ),
        )

        return improved

    async def _get_readability_score(self, content: str) -> float:
        """Get readability score for content"""

        try:
            flesch_score = await self.readability_scorer.calculate_flesch_reading_ease(
                content
            )
            return flesch_score / 100  # Normalize to 0-1
        except:
            return 0.5  # Default score

    def _initialize_transformation_strategies(
        self,
    ) -> Dict[TransformationType, callable]:
        """Initialize transformation strategies"""

        return {
            TransformationType.SIMPLIFY: self.simplify_content,
            TransformationType.ELABORATE: self.elaborate_content,
            TransformationType.ADJUST_TONE: self.adjust_tone,
            TransformationType.ADD_EXAMPLES: self._add_examples,
            TransformationType.REMOVE_JARGON: self._replace_jargon,
        }


class AdaptiveContentGenerator:
    """Main adaptive content generation system"""

    def __init__(self):
        self.prompt_framework = None
        self.audience_detector = AudienceDetector()
        self.content_analyzer = ContentAnalyzer()
        self.content_transformer = ContentTransformer()
        self._initialized = False

    async def initialize(self):
        """Initialize the adaptive content generation system"""
        if self._initialized:
            return

        logger.info("Initializing Adaptive Content Generation System")

        # Initialize prompt engineering framework
        self.prompt_framework = AdvancedPromptEngineeringFramework()
        await self.prompt_framework.initialize()

        self._initialized = True
        logger.info("Adaptive Content Generation System initialized successfully")

    async def generate_adaptive_content(
        self, request: ContentGenerationRequest
    ) -> GeneratedContent:
        """Generate content adapted to target audience"""

        start_time = asyncio.get_event_loop().time()
        content_id = str(uuid.uuid4())

        logger.info(
            "Generating adaptive content",
            topic=request.topic,
            audience_level=request.target_audience.level.value,
            content_type=request.content_type.value,
        )

        try:
            # Generate initial content using prompt framework
            initial_content = await self._generate_initial_content(request)

            # Analyze initial content
            content_analysis = await self.content_analyzer.analyze_content(
                initial_content
            )

            # Determine required transformations
            transformations = await self._determine_transformations(
                content_analysis, request.target_audience
            )

            # Apply transformations
            final_content = await self.content_transformer.transform_content(
                initial_content,
                content_analysis,
                request.target_audience,
                transformations,
            )

            # Optimize readability if needed
            if request.target_audience.level in [
                AudienceLevel.BEGINNER,
                AudienceLevel.K12_ELEMENTARY,
            ]:
                target_readability = 0.7  # High readability
                final_content, final_readability = (
                    await self.content_transformer.optimize_readability(
                        final_content, target_readability
                    )
                )
            else:
                final_readability = (
                    await self.content_transformer._get_readability_score(final_content)
                )

            # Final analysis
            final_analysis = await self.content_analyzer.analyze_content(final_content)

            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                final_analysis, request.target_audience, transformations
            )

            generation_time = asyncio.get_event_loop().time() - start_time

            result = GeneratedContent(
                content_id=content_id,
                content=final_content,
                metadata={
                    "topic": request.topic,
                    "audience_level": request.target_audience.level.value,
                    "content_type": request.content_type.value,
                    "initial_length": len(initial_content.split()),
                    "final_length": len(final_content.split()),
                    "transformations_count": len(transformations),
                },
                analysis=final_analysis,
                generation_time=generation_time,
                confidence_score=confidence_score,
                transformations_applied=transformations,
                readability_metrics={
                    "flesch_score": final_readability * 100,
                    "complexity_score": final_analysis.complexity_score,
                    "technical_density": final_analysis.technical_density,
                },
            )

            logger.info(
                "Adaptive content generated successfully",
                content_id=content_id,
                confidence=confidence_score,
                generation_time=generation_time,
                transformations=len(transformations),
            )

            return result

        except Exception as e:
            logger.error("Adaptive content generation failed", error=str(e))
            raise

    async def detect_and_adapt(
        self, topic: str, content_type: ContentType, context: Dict[str, Any]
    ) -> GeneratedContent:
        """Detect audience from context and generate adapted content"""

        # Detect audience profile
        audience_profile = await self.audience_detector.detect_audience(context)

        # Create generation request
        request = ContentGenerationRequest(
            topic=topic,
            target_audience=audience_profile,
            content_type=content_type,
            context=context,
        )

        return await self.generate_adaptive_content(request)

    async def validate_content_quality(
        self, content: str, target_audience: AudienceProfile
    ) -> Dict[str, float]:
        """Validate content quality against target audience requirements"""

        analysis = await self.content_analyzer.analyze_content(content)

        quality_metrics = {}

        # Readability appropriateness
        target_readability = self._get_target_readability(target_audience)
        readability_diff = abs(analysis.readability_score - target_readability)
        quality_metrics["readability_match"] = max(0, 1 - readability_diff)

        # Complexity appropriateness
        target_complexity = self._get_target_complexity(target_audience)
        complexity_diff = abs(analysis.complexity_score - target_complexity)
        quality_metrics["complexity_match"] = max(0, 1 - complexity_diff)

        # Technical density appropriateness
        target_technical_density = self._get_target_technical_density(target_audience)
        technical_diff = abs(analysis.technical_density - target_technical_density)
        quality_metrics["technical_density_match"] = max(0, 1 - technical_diff)

        # Overall quality score
        quality_metrics["overall_quality"] = (
            quality_metrics["readability_match"] * 0.4
            + quality_metrics["complexity_match"] * 0.3
            + quality_metrics["technical_density_match"] * 0.3
        )

        return quality_metrics

    async def _generate_initial_content(self, request: ContentGenerationRequest) -> str:
        """Generate initial content using prompt framework"""

        # Create prompt generation request
        prompt_request = PromptGenerationRequest(
            content_type=request.content_type,
            audience_level=request.target_audience.level,
            topic=request.topic,
            context=request.context,
            constraints=request.constraints,
            prompt_type=PromptType.FEW_SHOT,
        )

        # Generate prompt
        generated_prompt = await self.prompt_framework.generate_prompt(prompt_request)

        # For now, return a placeholder content
        # In a real implementation, this would call an LLM with the generated prompt
        placeholder_content = f"""
{request.topic}

This is a comprehensive explanation of {request.topic} tailored for {request.target_audience.level.value} level audience.

Key concepts include:
- Fundamental principles
- Practical applications
- Best practices
- Common challenges

The content is designed to be appropriate for the target audience's technical background and learning preferences.
        """.strip()

        return placeholder_content

    async def _determine_transformations(
        self, analysis: ContentAnalysis, target_audience: AudienceProfile
    ) -> List[TransformationType]:
        """Determine required transformations based on analysis and target audience"""

        transformations = []

        # Complexity adjustments
        target_complexity = self._get_target_complexity(target_audience)
        if analysis.complexity_score > target_complexity + 0.2:
            transformations.append(TransformationType.SIMPLIFY)
        elif analysis.complexity_score < target_complexity - 0.2:
            transformations.append(TransformationType.ELABORATE)

        # Tone adjustments
        target_tone = self._get_target_tone(target_audience)
        if analysis.tone != target_tone:
            transformations.append(TransformationType.ADJUST_TONE)

        # Jargon handling
        if len(analysis.jargon_terms) > 5 and target_audience.level in [
            AudienceLevel.BEGINNER,
            AudienceLevel.K12_ELEMENTARY,
        ]:
            transformations.append(TransformationType.REMOVE_JARGON)

        # Example additions
        if analysis.example_count < 2 and target_audience.level in [
            AudienceLevel.BEGINNER,
            AudienceLevel.INTERMEDIATE,
        ]:
            transformations.append(TransformationType.ADD_EXAMPLES)

        # Technical depth adjustments
        if (
            target_audience.level in [AudienceLevel.EXPERT, AudienceLevel.ADVANCED]
            and analysis.technical_density < 0.3
        ):
            transformations.append(TransformationType.INCREASE_TECHNICAL_DEPTH)

        return transformations

    async def _calculate_confidence_score(
        self,
        analysis: ContentAnalysis,
        target_audience: AudienceProfile,
        transformations: List[TransformationType],
    ) -> float:
        """Calculate confidence score for generated content"""

        base_score = 0.6

        # Quality metrics
        quality_metrics = await self.validate_content_quality(
            "", target_audience  # Content would be passed here in real implementation
        )
        base_score += quality_metrics["overall_quality"] * 0.3

        # Transformation success
        if transformations:
            transformation_bonus = min(len(transformations) * 0.05, 0.2)
            base_score += transformation_bonus

        # Content completeness
        if analysis.length > 50:  # Reasonable length
            base_score += 0.1

        return min(base_score, 1.0)

    def _get_target_readability(self, audience: AudienceProfile) -> float:
        """Get target readability score for audience"""

        readability_targets = {
            AudienceLevel.K12_ELEMENTARY: 0.9,
            AudienceLevel.K12_MIDDLE: 0.8,
            AudienceLevel.K12_HIGH: 0.7,
            AudienceLevel.BEGINNER: 0.7,
            AudienceLevel.INTERMEDIATE: 0.6,
            AudienceLevel.ADVANCED: 0.5,
            AudienceLevel.EXPERT: 0.4,
            AudienceLevel.UNDERGRADUATE: 0.6,
            AudienceLevel.GRADUATE: 0.5,
            AudienceLevel.PROFESSIONAL: 0.5,
        }

        return readability_targets.get(audience.level, 0.6)

    def _get_target_complexity(self, audience: AudienceProfile) -> float:
        """Get target complexity score for audience"""

        complexity_targets = {
            AudienceLevel.K12_ELEMENTARY: 0.2,
            AudienceLevel.K12_MIDDLE: 0.3,
            AudienceLevel.K12_HIGH: 0.4,
            AudienceLevel.BEGINNER: 0.3,
            AudienceLevel.INTERMEDIATE: 0.5,
            AudienceLevel.ADVANCED: 0.7,
            AudienceLevel.EXPERT: 0.9,
            AudienceLevel.UNDERGRADUATE: 0.5,
            AudienceLevel.GRADUATE: 0.7,
            AudienceLevel.PROFESSIONAL: 0.6,
        }

        return complexity_targets.get(audience.level, 0.5)

    def _get_target_technical_density(self, audience: AudienceProfile) -> float:
        """Get target technical density for audience"""

        if audience.technical_background:
            base_density = 0.4
        else:
            base_density = 0.2

        level_multipliers = {
            AudienceLevel.K12_ELEMENTARY: 0.1,
            AudienceLevel.K12_MIDDLE: 0.2,
            AudienceLevel.K12_HIGH: 0.3,
            AudienceLevel.BEGINNER: 0.3,
            AudienceLevel.INTERMEDIATE: 0.6,
            AudienceLevel.ADVANCED: 0.8,
            AudienceLevel.EXPERT: 1.0,
            AudienceLevel.UNDERGRADUATE: 0.5,
            AudienceLevel.GRADUATE: 0.8,
            AudienceLevel.PROFESSIONAL: 0.7,
        }

        multiplier = level_multipliers.get(audience.level, 0.5)
        return base_density * multiplier

    def _get_target_tone(self, audience: AudienceProfile) -> str:
        """Get target tone for audience"""

        if audience.level in [AudienceLevel.K12_ELEMENTARY, AudienceLevel.K12_MIDDLE]:
            return "conversational"
        elif audience.level in [AudienceLevel.GRADUATE, AudienceLevel.EXPERT]:
            return "academic"
        elif audience.level == AudienceLevel.PROFESSIONAL:
            return "formal"
        else:
            return "neutral"

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""

        return {
            "framework_initialized": self._initialized,
            "supported_audience_levels": len(AudienceLevel),
            "supported_content_types": len(ContentType),
            "supported_transformations": len(TransformationType),
            "supported_domains": len(ContentDomain),
        }

    async def shutdown(self):
        """Shutdown the adaptive content generation system"""
        logger.info("Shutting down Adaptive Content Generation System")

        if self.prompt_framework:
            await self.prompt_framework.shutdown()

        logger.info("Adaptive Content Generation System shutdown complete")


# Factory function
async def create_adaptive_content_generator() -> AdaptiveContentGenerator:
    """Create and initialize an adaptive content generator"""
    generator = AdaptiveContentGenerator()
    await generator.initialize()
    return generator
