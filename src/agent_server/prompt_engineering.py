"""
Advanced Prompt Engineering Framework
Implements sophisticated prompt engineering with few-shot and meta-prompting techniques
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import re
import uuid
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

from src.shared.logging import get_logger
from src.shared.config import get_settings

logger = get_logger(__name__)


class PromptType(Enum):
    """Types of prompts supported by the framework"""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    GENERATE_KNOWLEDGE = "generate_knowledge"
    META_PROMPT = "meta_prompt"
    SELF_CONSISTENCY = "self_consistency"
    REACT = "react"
    REFLECTION = "reflection"


class AudienceLevel(Enum):
    """Target audience complexity levels"""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    K12_ELEMENTARY = "k12_elementary"
    K12_MIDDLE = "k12_middle"
    K12_HIGH = "k12_high"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    PROFESSIONAL = "professional"


class ContentType(Enum):
    """Types of content that can be generated"""

    EXPLANATION = "explanation"
    CODE = "code"
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"
    QUIZ = "quiz"
    ASSIGNMENT = "assignment"
    ANALYSIS = "analysis"
    SUMMARY = "summary"
    COMPARISON = "comparison"


@dataclass
class PromptExample:
    """Represents an example for few-shot prompting"""

    input: str
    output: str
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class PromptTemplate:
    """Template for generating prompts"""

    template_id: str
    name: str
    template: str
    prompt_type: PromptType
    content_type: ContentType
    audience_level: Optional[AudienceLevel] = None
    variables: List[str] = field(default_factory=list)
    examples: List[PromptExample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    usage_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptGenerationRequest:
    """Request for prompt generation"""

    content_type: ContentType
    audience_level: AudienceLevel
    topic: str
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples_needed: int = 3
    prompt_type: PromptType = PromptType.FEW_SHOT
    quality_threshold: float = 0.8


@dataclass
class GeneratedPrompt:
    """Result of prompt generation"""

    prompt_id: str
    prompt_text: str
    prompt_type: PromptType
    examples_used: List[PromptExample]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    template_used: Optional[str] = None


class PromptOptimizer(ABC):
    """Abstract base class for prompt optimization strategies"""

    @abstractmethod
    async def optimize_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize a prompt for better performance"""
        pass

    @abstractmethod
    async def evaluate_prompt_quality(
        self, prompt: str, examples: List[PromptExample]
    ) -> float:
        """Evaluate the quality of a prompt"""
        pass


class FewShotOptimizer(PromptOptimizer):
    """Optimizer for few-shot prompts"""

    async def optimize_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize few-shot prompt by selecting best examples"""

        # Extract examples from context
        available_examples = context.get("available_examples", [])
        target_count = context.get("example_count", 3)

        if not available_examples:
            return prompt

        # Select best examples based on relevance and quality
        selected_examples = await self._select_best_examples(
            available_examples, target_count, context
        )

        # Reconstruct prompt with optimized examples
        optimized_prompt = await self._reconstruct_prompt_with_examples(
            prompt, selected_examples
        )

        return optimized_prompt

    async def evaluate_prompt_quality(
        self, prompt: str, examples: List[PromptExample]
    ) -> float:
        """Evaluate few-shot prompt quality"""

        if not examples:
            return 0.5  # Base score for zero-shot

        # Calculate quality based on example diversity and quality
        quality_scores = [ex.quality_score for ex in examples]
        avg_quality = sum(quality_scores) / len(quality_scores)

        # Bonus for diversity
        diversity_bonus = min(len(examples) * 0.1, 0.3)

        return min(avg_quality + diversity_bonus, 1.0)

    async def _select_best_examples(
        self,
        available_examples: List[PromptExample],
        target_count: int,
        context: Dict[str, Any],
    ) -> List[PromptExample]:
        """Select the best examples for few-shot prompting"""

        # Sort by quality score and success rate
        scored_examples = []
        for example in available_examples:
            score = example.quality_score * 0.7 + example.success_rate * 0.3
            scored_examples.append((score, example))

        # Sort by score descending
        scored_examples.sort(key=lambda x: x[0], reverse=True)

        # Select top examples ensuring diversity
        selected = []
        used_patterns = set()

        for score, example in scored_examples:
            if len(selected) >= target_count:
                break

            # Check for pattern diversity
            pattern = self._extract_pattern(example.input)
            if pattern not in used_patterns or len(selected) < target_count // 2:
                selected.append(example)
                used_patterns.add(pattern)

        return selected[:target_count]

    def _extract_pattern(self, text: str) -> str:
        """Extract pattern from example text for diversity checking"""
        # Simple pattern extraction based on first few words
        words = text.split()[:3]
        return " ".join(words).lower()

    async def _reconstruct_prompt_with_examples(
        self, base_prompt: str, examples: List[PromptExample]
    ) -> str:
        """Reconstruct prompt with selected examples"""

        examples_text = "\n\n".join(
            [
                f"Example {i+1}:\nInput: {ex.input}\nOutput: {ex.output}"
                for i, ex in enumerate(examples)
            ]
        )

        if "{{examples}}" in base_prompt:
            return base_prompt.replace("{{examples}}", examples_text)
        else:
            return f"{examples_text}\n\n{base_prompt}"


class MetaPromptOptimizer(PromptOptimizer):
    """Optimizer for meta-prompting"""

    async def optimize_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize prompt using meta-prompting techniques"""

        # Analyze the original prompt
        analysis = await self._analyze_prompt_structure(prompt)

        # Generate meta-prompt for optimization
        meta_prompt = await self._generate_meta_prompt(prompt, analysis, context)

        return meta_prompt

    async def evaluate_prompt_quality(
        self, prompt: str, examples: List[PromptExample]
    ) -> float:
        """Evaluate meta-prompt quality"""

        # Check for meta-prompting indicators
        meta_indicators = [
            "think step by step",
            "analyze the problem",
            "consider multiple approaches",
            "reflect on your answer",
            "verify your reasoning",
        ]

        indicator_count = sum(
            1 for indicator in meta_indicators if indicator in prompt.lower()
        )

        base_score = 0.6
        meta_bonus = min(indicator_count * 0.1, 0.4)

        return min(base_score + meta_bonus, 1.0)

    async def _analyze_prompt_structure(self, prompt: str) -> Dict[str, Any]:
        """Analyze the structure of a prompt"""

        analysis = {
            "length": len(prompt.split()),
            "has_examples": "example" in prompt.lower(),
            "has_instructions": any(
                word in prompt.lower() for word in ["please", "you should", "make sure"]
            ),
            "has_constraints": any(
                word in prompt.lower() for word in ["must", "should not", "avoid"]
            ),
            "complexity_level": self._assess_prompt_complexity(prompt),
        }

        return analysis

    def _assess_prompt_complexity(self, prompt: str) -> str:
        """Assess the complexity level of a prompt"""

        word_count = len(prompt.split())
        technical_terms = len(
            re.findall(r"\b(implement|algorithm|optimize|analyze)\b", prompt.lower())
        )

        if word_count > 200 or technical_terms > 5:
            return "high"
        elif word_count > 100 or technical_terms > 2:
            return "medium"
        else:
            return "low"

    async def _generate_meta_prompt(
        self, original_prompt: str, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate meta-prompt for optimization"""

        meta_instructions = []

        # Add reasoning instructions
        meta_instructions.append(
            "Before providing your final answer, think through the problem step by step."
        )

        # Add verification instructions
        if analysis["complexity_level"] in ["medium", "high"]:
            meta_instructions.append(
                "After generating your response, review it for accuracy and completeness."
            )

        # Add context-specific instructions
        if context.get("audience_level") == AudienceLevel.BEGINNER:
            meta_instructions.append(
                "Explain concepts clearly and avoid unnecessary jargon."
            )

        # Combine with original prompt
        meta_prompt = "\n".join(meta_instructions) + "\n\n" + original_prompt

        return meta_prompt


class AdvancedPromptEngineeringFramework:
    """Main framework for advanced prompt engineering"""

    def __init__(self):
        self.settings = get_settings()
        self.templates: Dict[str, PromptTemplate] = {}
        self.examples_database: Dict[str, List[PromptExample]] = {}
        self.optimizers: Dict[PromptType, PromptOptimizer] = {}
        self.usage_analytics: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the prompt engineering framework"""
        if self._initialized:
            return

        logger.info("Initializing Advanced Prompt Engineering Framework")

        # Initialize optimizers
        self.optimizers[PromptType.FEW_SHOT] = FewShotOptimizer()
        self.optimizers[PromptType.META_PROMPT] = MetaPromptOptimizer()

        # Load default templates
        await self._load_default_templates()

        # Load examples database
        await self._load_examples_database()

        # Initialize analytics
        await self._initialize_analytics()

        self._initialized = True
        logger.info("Advanced Prompt Engineering Framework initialized successfully")

    async def generate_prompt(
        self, request: PromptGenerationRequest
    ) -> GeneratedPrompt:
        """Generate an optimized prompt based on the request"""

        start_time = asyncio.get_event_loop().time()
        prompt_id = str(uuid.uuid4())

        logger.info(
            "Generating prompt",
            content_type=request.content_type.value,
            audience_level=request.audience_level.value,
            prompt_type=request.prompt_type.value,
        )

        try:
            # Find appropriate template
            template = await self._find_best_template(request)

            # Generate base prompt
            base_prompt = await self._generate_base_prompt(template, request)

            # Select examples if needed
            examples = []
            if request.prompt_type in [
                PromptType.FEW_SHOT,
                PromptType.CHAIN_OF_THOUGHT,
            ]:
                examples = await self._select_examples(request)

            # Apply optimization
            optimized_prompt = await self._optimize_prompt(
                base_prompt, request.prompt_type, examples, request.context
            )

            # Calculate confidence score
            confidence = await self._calculate_confidence_score(
                optimized_prompt, examples, request
            )

            generation_time = asyncio.get_event_loop().time() - start_time

            result = GeneratedPrompt(
                prompt_id=prompt_id,
                prompt_text=optimized_prompt,
                prompt_type=request.prompt_type,
                examples_used=examples,
                confidence_score=confidence,
                metadata={
                    "template_id": template.template_id if template else None,
                    "audience_level": request.audience_level.value,
                    "content_type": request.content_type.value,
                    "topic": request.topic,
                },
                generation_time=generation_time,
                template_used=template.template_id if template else None,
            )

            # Update analytics
            await self._update_usage_analytics(result, request)

            logger.info(
                "Prompt generated successfully",
                prompt_id=prompt_id,
                confidence=confidence,
                generation_time=generation_time,
            )

            return result

        except Exception as e:
            logger.error("Prompt generation failed", error=str(e))
            raise

    async def create_few_shot_prompt(
        self,
        topic: str,
        audience_level: AudienceLevel,
        content_type: ContentType,
        example_count: int = 3,
    ) -> GeneratedPrompt:
        """Create a few-shot prompt with dynamic example selection"""

        request = PromptGenerationRequest(
            content_type=content_type,
            audience_level=audience_level,
            topic=topic,
            examples_needed=example_count,
            prompt_type=PromptType.FEW_SHOT,
        )

        return await self.generate_prompt(request)

    async def create_generate_knowledge_prompt(
        self, topic: str, audience_level: AudienceLevel
    ) -> GeneratedPrompt:
        """Create a Generate Knowledge Prompting prompt for technical accuracy"""

        request = PromptGenerationRequest(
            content_type=ContentType.EXPLANATION,
            audience_level=audience_level,
            topic=topic,
            prompt_type=PromptType.GENERATE_KNOWLEDGE,
            context={"require_technical_accuracy": True},
        )

        return await self.generate_prompt(request)

    async def create_meta_prompt(
        self, base_prompt: str, optimization_goals: List[str]
    ) -> GeneratedPrompt:
        """Create meta-prompt for query refinement and optimization"""

        request = PromptGenerationRequest(
            content_type=ContentType.ANALYSIS,
            audience_level=AudienceLevel.EXPERT,
            topic="prompt optimization",
            prompt_type=PromptType.META_PROMPT,
            context={
                "base_prompt": base_prompt,
                "optimization_goals": optimization_goals,
            },
        )

        return await self.generate_prompt(request)

    async def add_template(self, template: PromptTemplate) -> bool:
        """Add a new prompt template"""

        try:
            # Validate template
            if not await self._validate_template(template):
                return False

            # Store template
            self.templates[template.template_id] = template

            logger.info("Template added successfully", template_id=template.template_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to add template", template_id=template.template_id, error=str(e)
            )
            return False

    async def add_example(
        self, content_type: ContentType, example: PromptExample
    ) -> bool:
        """Add a new example to the database"""

        try:
            key = content_type.value
            if key not in self.examples_database:
                self.examples_database[key] = []

            self.examples_database[key].append(example)

            logger.info("Example added successfully", content_type=content_type.value)
            return True

        except Exception as e:
            logger.error("Failed to add example", error=str(e))
            return False

    async def evaluate_prompt_performance(
        self, prompt_id: str, success: bool, quality_score: float
    ) -> bool:
        """Evaluate and record prompt performance"""

        try:
            # Update analytics
            if prompt_id not in self.usage_analytics:
                self.usage_analytics[prompt_id] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "total_quality_score": 0.0,
                    "average_quality": 0.0,
                }

            analytics = self.usage_analytics[prompt_id]
            analytics["usage_count"] += 1

            if success:
                analytics["success_count"] += 1

            analytics["total_quality_score"] += quality_score
            analytics["average_quality"] = (
                analytics["total_quality_score"] / analytics["usage_count"]
            )

            logger.info(
                "Prompt performance recorded",
                prompt_id=prompt_id,
                success=success,
                quality_score=quality_score,
            )

            return True

        except Exception as e:
            logger.error("Failed to record prompt performance", error=str(e))
            return False

    async def get_template_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all templates"""

        metrics = {}

        for template_id, template in self.templates.items():
            usage_stats = template.usage_stats

            metrics[template_id] = {
                "name": template.name,
                "prompt_type": template.prompt_type.value,
                "content_type": template.content_type.value,
                "usage_count": usage_stats.get("usage_count", 0),
                "success_rate": usage_stats.get("success_rate", 0.0),
                "average_quality": usage_stats.get("average_quality", 0.0),
                "version": template.version,
            }

        return metrics

    async def optimize_existing_prompt(
        self, prompt_text: str, prompt_type: PromptType, context: Dict[str, Any]
    ) -> str:
        """Optimize an existing prompt"""

        if prompt_type in self.optimizers:
            optimizer = self.optimizers[prompt_type]
            return await optimizer.optimize_prompt(prompt_text, context)

        return prompt_text

    async def _find_best_template(
        self, request: PromptGenerationRequest
    ) -> Optional[PromptTemplate]:
        """Find the best template for the request"""

        candidates = []

        for template in self.templates.values():
            score = 0

            # Content type match
            if template.content_type == request.content_type:
                score += 3

            # Prompt type match
            if template.prompt_type == request.prompt_type:
                score += 2

            # Audience level match
            if template.audience_level == request.audience_level:
                score += 1

            # Performance-based scoring
            usage_stats = template.usage_stats
            if usage_stats.get("success_rate", 0) > 0.8:
                score += 1

            if score > 0:
                candidates.append((score, template))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    async def _generate_base_prompt(
        self, template: Optional[PromptTemplate], request: PromptGenerationRequest
    ) -> str:
        """Generate base prompt from template"""

        if template:
            # Use template
            prompt = template.template

            # Replace variables
            variables = {
                "topic": request.topic,
                "audience_level": request.audience_level.value,
                "content_type": request.content_type.value,
            }

            for var, value in variables.items():
                prompt = prompt.replace(f"{{{{{var}}}}}", str(value))

            return prompt
        else:
            # Generate default prompt
            return await self._generate_default_prompt(request)

    async def _generate_default_prompt(self, request: PromptGenerationRequest) -> str:
        """Generate default prompt when no template is available"""

        audience_instructions = {
            AudienceLevel.BEGINNER: "Explain in simple terms suitable for beginners.",
            AudienceLevel.INTERMEDIATE: "Provide a balanced explanation with some technical details.",
            AudienceLevel.ADVANCED: "Include advanced concepts and technical depth.",
            AudienceLevel.EXPERT: "Provide expert-level analysis with comprehensive technical details.",
            AudienceLevel.K12_ELEMENTARY: "Explain in very simple terms suitable for elementary students.",
            AudienceLevel.K12_MIDDLE: "Use age-appropriate language for middle school students.",
            AudienceLevel.K12_HIGH: "Provide explanations suitable for high school students.",
            AudienceLevel.UNDERGRADUATE: "Include university-level concepts and explanations.",
            AudienceLevel.GRADUATE: "Provide graduate-level depth and analysis.",
            AudienceLevel.PROFESSIONAL: "Focus on practical applications and professional context.",
        }

        content_instructions = {
            ContentType.EXPLANATION: f"Provide a clear explanation of {request.topic}.",
            ContentType.CODE: f"Generate code related to {request.topic}.",
            ContentType.DOCUMENTATION: f"Create documentation for {request.topic}.",
            ContentType.TUTORIAL: f"Create a tutorial about {request.topic}.",
            ContentType.QUIZ: f"Generate quiz questions about {request.topic}.",
            ContentType.ASSIGNMENT: f"Create an assignment related to {request.topic}.",
            ContentType.ANALYSIS: f"Analyze {request.topic}.",
            ContentType.SUMMARY: f"Summarize {request.topic}.",
            ContentType.COMPARISON: f"Compare different aspects of {request.topic}.",
        }

        base_instruction = content_instructions.get(
            request.content_type, f"Provide information about {request.topic}."
        )

        audience_instruction = audience_instructions.get(
            request.audience_level, "Provide an appropriate explanation."
        )

        return f"{base_instruction} {audience_instruction}"

    async def _select_examples(
        self, request: PromptGenerationRequest
    ) -> List[PromptExample]:
        """Select examples for few-shot prompting"""

        key = request.content_type.value
        available_examples = self.examples_database.get(key, [])

        if not available_examples:
            return []

        # Filter examples by quality threshold
        quality_examples = [
            ex
            for ex in available_examples
            if ex.quality_score >= request.quality_threshold
        ]

        if not quality_examples:
            quality_examples = available_examples

        # Sort by relevance and quality
        scored_examples = []
        for example in quality_examples:
            relevance_score = await self._calculate_example_relevance(
                example, request.topic, request.context
            )
            combined_score = example.quality_score * 0.6 + relevance_score * 0.4
            scored_examples.append((combined_score, example))

        # Sort and select top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        selected = [ex for _, ex in scored_examples[: request.examples_needed]]

        return selected

    async def _calculate_example_relevance(
        self, example: PromptExample, topic: str, context: Dict[str, Any]
    ) -> float:
        """Calculate relevance of an example to the current request"""

        # Simple keyword-based relevance (can be enhanced with embeddings)
        topic_words = set(topic.lower().split())
        example_words = set((example.input + " " + example.output).lower().split())

        common_words = topic_words.intersection(example_words)
        relevance = len(common_words) / max(len(topic_words), 1)

        return min(relevance, 1.0)

    async def _optimize_prompt(
        self,
        base_prompt: str,
        prompt_type: PromptType,
        examples: List[PromptExample],
        context: Dict[str, Any],
    ) -> str:
        """Apply optimization to the prompt"""

        if prompt_type in self.optimizers:
            optimizer = self.optimizers[prompt_type]
            optimization_context = {
                **context,
                "available_examples": examples,
                "example_count": len(examples),
            }
            return await optimizer.optimize_prompt(base_prompt, optimization_context)

        # Default optimization: add examples if available
        if examples and prompt_type == PromptType.FEW_SHOT:
            examples_text = "\n\n".join(
                [
                    f"Example {i+1}:\nInput: {ex.input}\nOutput: {ex.output}"
                    for i, ex in enumerate(examples)
                ]
            )
            return f"{examples_text}\n\n{base_prompt}"

        return base_prompt

    async def _calculate_confidence_score(
        self,
        prompt: str,
        examples: List[PromptExample],
        request: PromptGenerationRequest,
    ) -> float:
        """Calculate confidence score for the generated prompt"""

        base_score = 0.5

        # Template usage bonus
        if request.prompt_type != PromptType.ZERO_SHOT:
            base_score += 0.1

        # Examples quality bonus
        if examples:
            avg_example_quality = sum(ex.quality_score for ex in examples) / len(
                examples
            )
            base_score += avg_example_quality * 0.2

        # Prompt length and structure bonus
        word_count = len(prompt.split())
        if 50 <= word_count <= 200:  # Optimal length range
            base_score += 0.1

        # Audience-specific adjustments
        if request.audience_level in [AudienceLevel.EXPERT, AudienceLevel.PROFESSIONAL]:
            base_score += 0.05

        return min(base_score, 1.0)

    async def _update_usage_analytics(
        self, result: GeneratedPrompt, request: PromptGenerationRequest
    ):
        """Update usage analytics for templates and examples"""

        # Update template usage
        if result.template_used:
            template = self.templates.get(result.template_used)
            if template:
                if "usage_count" not in template.usage_stats:
                    template.usage_stats["usage_count"] = 0
                template.usage_stats["usage_count"] += 1

        # Update example usage
        for example in result.examples_used:
            example.usage_count += 1

    async def _validate_template(self, template: PromptTemplate) -> bool:
        """Validate a prompt template"""

        if not template.template_id or not template.name or not template.template:
            return False

        # Check for required variables
        required_vars = ["topic", "audience_level", "content_type"]
        for var in required_vars:
            if f"{{{{{var}}}}}" not in template.template and var in template.variables:
                logger.warning(
                    "Template missing required variable",
                    template_id=template.template_id,
                    variable=var,
                )

        return True

    async def _load_default_templates(self):
        """Load default prompt templates"""

        # Few-shot explanation template
        explanation_template = PromptTemplate(
            template_id="few_shot_explanation",
            name="Few-Shot Explanation Template",
            template="""{{examples}}

Now, provide a {{audience_level}} level explanation of {{topic}}. {{content_type}} should be clear, accurate, and appropriate for the target audience.""",
            prompt_type=PromptType.FEW_SHOT,
            content_type=ContentType.EXPLANATION,
            variables=["examples", "audience_level", "topic", "content_type"],
        )

        # Code generation template
        code_template = PromptTemplate(
            template_id="few_shot_code",
            name="Few-Shot Code Generation Template",
            template="""{{examples}}

Generate {{content_type}} code for {{topic}}. The code should be:
- Well-documented and commented
- Follow best practices for the programming language
- Be appropriate for {{audience_level}} level developers
- Include error handling where appropriate""",
            prompt_type=PromptType.FEW_SHOT,
            content_type=ContentType.CODE,
            variables=["examples", "content_type", "topic", "audience_level"],
        )

        # Generate Knowledge template
        knowledge_template = PromptTemplate(
            template_id="generate_knowledge",
            name="Generate Knowledge Prompting Template",
            template="""First, let me gather relevant knowledge about {{topic}}:

1. What are the key concepts and principles related to {{topic}}?
2. What are the current best practices and standards?
3. What are common challenges or misconceptions?

Based on this knowledge, I will now provide a {{audience_level}} level {{content_type}} about {{topic}}.""",
            prompt_type=PromptType.GENERATE_KNOWLEDGE,
            content_type=ContentType.EXPLANATION,
            variables=["topic", "audience_level", "content_type"],
        )

        # Meta-prompt template
        meta_template = PromptTemplate(
            template_id="meta_prompt",
            name="Meta-Prompting Template",
            template="""I need to optimize the following prompt for better results:

Original prompt: {{base_prompt}}

Optimization goals: {{optimization_goals}}

Let me analyze this prompt step by step:
1. What is the main objective?
2. What information might be missing?
3. How can I make the instructions clearer?
4. What examples or constraints should I add?

Based on this analysis, here is the optimized prompt:""",
            prompt_type=PromptType.META_PROMPT,
            content_type=ContentType.ANALYSIS,
            variables=["base_prompt", "optimization_goals"],
        )

        # Store templates
        templates = [
            explanation_template,
            code_template,
            knowledge_template,
            meta_template,
        ]
        for template in templates:
            self.templates[template.template_id] = template

        logger.info("Default templates loaded", count=len(templates))

    async def _load_examples_database(self):
        """Load default examples database"""

        # Code examples
        code_examples = [
            PromptExample(
                input="Create a Python function to calculate factorial",
                output="""def factorial(n):
    \"\"\"Calculate factorial of a non-negative integer.
    
    Args:
        n (int): Non-negative integer
        
    Returns:
        int: Factorial of n
        
    Raises:
        ValueError: If n is negative
    \"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)""",
                explanation="Well-documented recursive factorial function with error handling",
                quality_score=0.9,
            ),
            PromptExample(
                input="Implement a binary search algorithm in Python",
                output="""def binary_search(arr, target):
    \"\"\"Perform binary search on a sorted array.
    
    Args:
        arr (list): Sorted list of comparable elements
        target: Element to search for
        
    Returns:
        int: Index of target if found, -1 otherwise
    \"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
                explanation="Efficient binary search implementation with clear documentation",
                quality_score=0.95,
            ),
        ]

        # Explanation examples
        explanation_examples = [
            PromptExample(
                input="Explain object-oriented programming for beginners",
                output="""Object-Oriented Programming (OOP) is a programming approach that organizes code around "objects" - think of them as digital containers that hold both data and the actions that can be performed on that data.

Key Concepts:
1. **Class**: A blueprint or template (like a cookie cutter)
2. **Object**: An actual instance created from a class (like a cookie made from the cutter)
3. **Encapsulation**: Keeping related data and functions together
4. **Inheritance**: Creating new classes based on existing ones
5. **Polymorphism**: Using the same interface for different types of objects

Example: A "Car" class might have data like color and speed, and actions like start() and stop().""",
                explanation="Clear beginner-friendly explanation with analogies",
                quality_score=0.85,
            )
        ]

        # Store examples
        self.examples_database[ContentType.CODE.value] = code_examples
        self.examples_database[ContentType.EXPLANATION.value] = explanation_examples

        logger.info(
            "Examples database loaded",
            code_examples=len(code_examples),
            explanation_examples=len(explanation_examples),
        )

    async def _initialize_analytics(self):
        """Initialize analytics tracking"""

        self.usage_analytics = {}
        logger.info("Analytics initialized")

    async def get_framework_metrics(self) -> Dict[str, Any]:
        """Get framework performance metrics"""

        total_templates = len(self.templates)
        total_examples = sum(
            len(examples) for examples in self.examples_database.values()
        )

        # Calculate average template performance
        template_performances = []
        for template in self.templates.values():
            usage_stats = template.usage_stats
            if usage_stats.get("usage_count", 0) > 0:
                template_performances.append(usage_stats.get("success_rate", 0.0))

        avg_template_performance = (
            sum(template_performances) / len(template_performances)
            if template_performances
            else 0.0
        )

        return {
            "total_templates": total_templates,
            "total_examples": total_examples,
            "average_template_performance": avg_template_performance,
            "prompt_types_supported": len(PromptType),
            "content_types_supported": len(ContentType),
            "audience_levels_supported": len(AudienceLevel),
            "optimizers_available": len(self.optimizers),
        }

    async def export_templates(self) -> Dict[str, Any]:
        """Export all templates for backup or sharing"""

        exported = {}
        for template_id, template in self.templates.items():
            exported[template_id] = {
                "name": template.name,
                "template": template.template,
                "prompt_type": template.prompt_type.value,
                "content_type": template.content_type.value,
                "audience_level": (
                    template.audience_level.value if template.audience_level else None
                ),
                "variables": template.variables,
                "version": template.version,
                "usage_stats": template.usage_stats,
            }

        return exported

    async def import_templates(self, templates_data: Dict[str, Any]) -> int:
        """Import templates from exported data"""

        imported_count = 0

        for template_id, data in templates_data.items():
            try:
                template = PromptTemplate(
                    template_id=template_id,
                    name=data["name"],
                    template=data["template"],
                    prompt_type=PromptType(data["prompt_type"]),
                    content_type=ContentType(data["content_type"]),
                    audience_level=(
                        AudienceLevel(data["audience_level"])
                        if data.get("audience_level")
                        else None
                    ),
                    variables=data.get("variables", []),
                    version=data.get("version", "1.0"),
                    usage_stats=data.get("usage_stats", {}),
                )

                if await self.add_template(template):
                    imported_count += 1

            except Exception as e:
                logger.error(
                    "Failed to import template", template_id=template_id, error=str(e)
                )

        logger.info("Templates imported", count=imported_count)
        return imported_count

    async def shutdown(self):
        """Shutdown the framework and cleanup resources"""
        logger.info("Shutting down Advanced Prompt Engineering Framework")

        # Save analytics and usage data if needed
        # This could be extended to persist data to a database

        logger.info("Advanced Prompt Engineering Framework shutdown complete")


# Factory functions for common prompt types
async def create_few_shot_prompt_framework() -> AdvancedPromptEngineeringFramework:
    """Create and initialize a prompt engineering framework"""
    framework = AdvancedPromptEngineeringFramework()
    await framework.initialize()
    return framework


async def generate_educational_content_prompt(
    topic: str, grade_level: str, content_type: str
) -> GeneratedPrompt:
    """Generate prompt for educational content creation"""

    framework = await create_few_shot_prompt_framework()

    # Map grade level to audience level
    grade_mapping = {
        "elementary": AudienceLevel.K12_ELEMENTARY,
        "middle": AudienceLevel.K12_MIDDLE,
        "high": AudienceLevel.K12_HIGH,
        "undergraduate": AudienceLevel.UNDERGRADUATE,
        "graduate": AudienceLevel.GRADUATE,
    }

    # Map content type
    content_mapping = {
        "explanation": ContentType.EXPLANATION,
        "tutorial": ContentType.TUTORIAL,
        "quiz": ContentType.QUIZ,
        "assignment": ContentType.ASSIGNMENT,
    }

    audience = grade_mapping.get(grade_level.lower(), AudienceLevel.INTERMEDIATE)
    content = content_mapping.get(content_type.lower(), ContentType.EXPLANATION)

    return await framework.create_few_shot_prompt(topic, audience, content)


async def generate_code_prompt(
    language: str, task: str, skill_level: str
) -> GeneratedPrompt:
    """Generate prompt for code generation"""

    framework = await create_few_shot_prompt_framework()

    # Map skill level to audience level
    skill_mapping = {
        "beginner": AudienceLevel.BEGINNER,
        "intermediate": AudienceLevel.INTERMEDIATE,
        "advanced": AudienceLevel.ADVANCED,
        "expert": AudienceLevel.EXPERT,
    }

    audience = skill_mapping.get(skill_level.lower(), AudienceLevel.INTERMEDIATE)
    topic = f"{task} in {language}"

    return await framework.create_few_shot_prompt(topic, audience, ContentType.CODE)
