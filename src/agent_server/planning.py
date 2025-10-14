"""
Planning Module
Advanced task decomposition and planning capabilities
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid
import re
from enum import Enum

from .orchestrator import ExecutionPlan
from src.shared.logging import get_logger

logger = get_logger(__name__)


class IntentType(Enum):
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CONTENT_GENERATION = "content_generation"
    CODE_GENERATION = "code_generation"
    DOCUMENT_GENERATION = "document_generation"
    ANALYSIS = "analysis"
    MULTI_STEP = "multi_step"
    GENERAL_QUERY = "general_query"


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class EntityType(Enum):
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    CONCEPT = "concept"
    FILE_FORMAT = "file_format"
    TOOL = "tool"
    PERSON = "person"
    ORGANIZATION = "organization"


@dataclass
class ConversationContext:
    session_id: str
    user_id: Optional[str]
    message_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_topic: Optional[str] = None
    domain_context: Dict[str, Any] = field(default_factory=dict)
    conversation_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    text: str
    entity_type: EntityType
    confidence: float
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    intent: IntentType
    entities: List[Entity]
    complexity: ComplexityLevel
    domain: str
    confidence: float
    keywords: List[str] = field(default_factory=list)
    semantic_features: Dict[str, Any] = field(default_factory=dict)
    user_goals: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskTemplate:
    task_type: str
    required_inputs: List[str]
    optional_inputs: List[str]
    expected_outputs: List[str]
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    failure_modes: List[str]
    recovery_strategies: List[str]


@dataclass
class Goal:
    goal_id: str
    description: str
    priority: int
    success_criteria: List[str]
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_effort: float = 1.0


class PlanningModule:
    """Advanced planning module with hierarchical task decomposition"""

    def __init__(self):
        self._initialized = False
        self.intent_patterns = self._initialize_intent_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
        self.task_templates = self._initialize_task_templates()
        self.complexity_indicators = self._initialize_complexity_indicators()
        self.domain_keywords = self._initialize_domain_keywords()
        self.planning_strategies = self._initialize_planning_strategies()

    async def initialize(self):
        """Initialize planning components"""
        if self._initialized:
            return

        logger.info("Initializing Planning Module")

        # Initialize advanced planning components
        await self._initialize_goal_decomposition()
        await self._initialize_dependency_analyzer()
        await self._initialize_adaptation_engine()

        self._initialized = True
        logger.info("Planning Module initialized with advanced capabilities")

    async def create_plan(
        self, message: str, context: ConversationContext
    ) -> ExecutionPlan:
        """Create comprehensive execution plan from user message"""
        logger.info(
            "Creating comprehensive execution plan", session_id=context.session_id
        )

        # Step 1: Comprehensive query analysis
        analysis = await self._analyze_query_comprehensive(message, context)

        # Step 2: Goal identification and prioritization
        goals = await self._identify_and_prioritize_goals(analysis, context)

        # Step 3: Hierarchical task decomposition
        tasks = await self._hierarchical_task_decomposition(goals, analysis, context)

        # Step 4: Advanced dependency analysis
        dependencies = await self._advanced_dependency_analysis(tasks, goals)

        # Step 5: Resource estimation and optimization
        estimated_duration = await self._estimate_duration_advanced(tasks, dependencies)

        # Step 6: Recovery strategy planning
        recovery_strategies = await self._plan_recovery_strategies(tasks, analysis)

        # Step 7: Parallel execution optimization
        parallel_groups = await self._identify_parallel_execution_groups(
            tasks, dependencies
        )

        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            tasks=tasks,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            priority=self._determine_priority_advanced(analysis, goals),
            recovery_strategies=recovery_strategies,
            parallel_groups=parallel_groups,
        )

        logger.info(
            "Comprehensive execution plan created",
            plan_id=plan.plan_id,
            task_count=len(tasks),
            goal_count=len(goals),
            estimated_duration=estimated_duration,
            parallel_groups=len(parallel_groups),
        )

        return plan

    async def _analyze_query_comprehensive(
        self, message: str, context: ConversationContext
    ) -> QueryAnalysis:
        """Comprehensive query analysis with intent classification and entity extraction"""

        # Intent classification using pattern matching and context
        intent = await self._classify_intent_advanced(message, context)

        # Entity extraction
        entities = await self._extract_entities_advanced(message, context)

        # Complexity assessment
        complexity = await self._assess_complexity_advanced(message, context, entities)

        # Keyword extraction
        keywords = await self._extract_keywords(message)

        # Semantic feature extraction
        semantic_features = await self._extract_semantic_features(message, context)

        # Goal identification
        user_goals = await self._identify_user_goals(message, context, entities)

        # Constraint identification
        constraints = await self._identify_constraints(message, context)

        # Domain classification
        domain = await self._classify_domain(message, entities, keywords)

        # Confidence calculation
        confidence = await self._calculate_analysis_confidence(
            intent, entities, complexity
        )

        return QueryAnalysis(
            intent=intent,
            entities=entities,
            complexity=complexity,
            domain=domain,
            confidence=confidence,
            keywords=keywords,
            semantic_features=semantic_features,
            user_goals=user_goals,
            constraints=constraints,
        )

    async def _classify_intent_advanced(
        self, message: str, context: ConversationContext
    ) -> IntentType:
        """Advanced intent classification using patterns and context"""

        message_lower = message.lower()

        # Multi-step detection
        multi_step_indicators = [
            "and then",
            "after that",
            "next",
            "also",
            "additionally",
            "furthermore",
        ]
        if any(indicator in message_lower for indicator in multi_step_indicators):
            return IntentType.MULTI_STEP

        # Document generation detection
        doc_patterns = [
            "generate document",
            "create report",
            "write documentation",
            "export to",
            "save as",
        ]
        if any(pattern in message_lower for pattern in doc_patterns):
            return IntentType.DOCUMENT_GENERATION

        # Code generation detection
        code_patterns = [
            "write code",
            "implement",
            "create function",
            "build class",
            "develop",
            "program",
        ]
        if any(pattern in message_lower for pattern in code_patterns):
            return IntentType.CODE_GENERATION

        # Content generation detection
        content_patterns = ["generate", "create content", "write", "compose", "draft"]
        if any(pattern in message_lower for pattern in content_patterns):
            return IntentType.CONTENT_GENERATION

        # Analysis detection
        analysis_patterns = [
            "analyze",
            "review",
            "evaluate",
            "assess",
            "compare",
            "examine",
        ]
        if any(pattern in message_lower for pattern in analysis_patterns):
            return IntentType.ANALYSIS

        # Knowledge retrieval detection
        knowledge_patterns = [
            "explain",
            "what is",
            "how does",
            "why",
            "tell me about",
            "describe",
        ]
        if any(pattern in message_lower for pattern in knowledge_patterns):
            return IntentType.KNOWLEDGE_RETRIEVAL

        # Context-based intent refinement
        if context.current_topic:
            if context.current_topic == "programming" and any(
                word in message_lower for word in ["help", "show", "example"]
            ):
                return IntentType.CODE_GENERATION

        return IntentType.GENERAL_QUERY

    async def _extract_entities_advanced(
        self, message: str, context: ConversationContext
    ) -> List[Entity]:
        """Advanced entity extraction using patterns and context"""

        entities = []

        # Programming language detection
        prog_languages = [
            "python",
            "javascript",
            "java",
            "c++",
            "c#",
            "go",
            "rust",
            "typescript",
            "php",
            "ruby",
        ]
        for lang in prog_languages:
            if lang in message.lower():
                entities.append(
                    Entity(
                        text=lang,
                        entity_type=EntityType.PROGRAMMING_LANGUAGE,
                        confidence=0.9,
                        context="programming language mentioned",
                    )
                )

        # Framework detection
        frameworks = [
            "react",
            "angular",
            "vue",
            "django",
            "flask",
            "spring",
            "express",
            "fastapi",
        ]
        for framework in frameworks:
            if framework in message.lower():
                entities.append(
                    Entity(
                        text=framework,
                        entity_type=EntityType.FRAMEWORK,
                        confidence=0.85,
                        context="framework mentioned",
                    )
                )

        # File format detection
        file_formats = ["pdf", "docx", "pptx", "json", "xml", "csv", "yaml", "markdown"]
        for fmt in file_formats:
            if fmt in message.lower():
                entities.append(
                    Entity(
                        text=fmt,
                        entity_type=EntityType.FILE_FORMAT,
                        confidence=0.8,
                        context="file format mentioned",
                    )
                )

        # Tool detection
        tools = [
            "git",
            "docker",
            "kubernetes",
            "jenkins",
            "github",
            "gitlab",
            "aws",
            "azure",
        ]
        for tool in tools:
            if tool in message.lower():
                entities.append(
                    Entity(
                        text=tool,
                        entity_type=EntityType.TOOL,
                        confidence=0.8,
                        context="tool mentioned",
                    )
                )

        # Concept detection using regex patterns
        concept_patterns = [
            (
                r"\b(design pattern|algorithm|data structure|architecture)\b",
                EntityType.CONCEPT,
            ),
            (r"\b(testing|unit test|integration test|debugging)\b", EntityType.CONCEPT),
            (r"\b(api|rest|graphql|microservice)\b", EntityType.CONCEPT),
        ]

        for pattern, entity_type in concept_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                entities.append(
                    Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        confidence=0.75,
                        context="concept pattern match",
                    )
                )

        return entities

    async def _assess_complexity_advanced(
        self, message: str, context: ConversationContext, entities: List[Entity]
    ) -> ComplexityLevel:
        """Advanced complexity assessment"""

        complexity_score = 0

        # Length-based scoring
        word_count = len(message.split())
        if word_count > 100:
            complexity_score += 3
        elif word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1

        # Entity-based scoring
        complexity_score += len(entities) * 0.5

        # Multi-step indicators
        multi_step_words = [
            "and",
            "then",
            "after",
            "also",
            "additionally",
            "furthermore",
            "moreover",
        ]
        complexity_score += sum(
            1 for word in multi_step_words if word in message.lower()
        )

        # Technical complexity indicators
        technical_indicators = [
            "integrate",
            "optimize",
            "scale",
            "deploy",
            "configure",
            "customize",
        ]
        complexity_score += (
            sum(1 for indicator in technical_indicators if indicator in message.lower())
            * 0.5
        )

        # Context-based complexity
        if context.current_topic and context.current_topic in [
            "programming",
            "architecture",
        ]:
            complexity_score += 1

        # Determine complexity level
        if complexity_score >= 6:
            return ComplexityLevel.VERY_COMPLEX
        elif complexity_score >= 4:
            return ComplexityLevel.COMPLEX
        elif complexity_score >= 2:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

    async def _extract_keywords(self, message: str) -> List[str]:
        """Extract important keywords from the message"""

        # Simple keyword extraction (can be enhanced with NLP libraries)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }

        words = re.findall(r"\b\w+\b", message.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Return top keywords by frequency
        from collections import Counter

        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]

    async def _extract_semantic_features(
        self, message: str, context: ConversationContext
    ) -> Dict[str, Any]:
        """Extract semantic features from the message"""

        features = {
            "question_words": len(
                re.findall(r"\b(what|how|why|when|where|who)\b", message.lower())
            ),
            "imperative_verbs": len(
                re.findall(
                    r"\b(create|generate|build|make|write|implement)\b", message.lower()
                )
            ),
            "technical_terms": len(
                re.findall(
                    r"\b(api|database|server|client|framework|library)\b",
                    message.lower(),
                )
            ),
            "urgency_indicators": len(
                re.findall(
                    r"\b(urgent|asap|quickly|immediately|now)\b", message.lower()
                )
            ),
            "quality_indicators": len(
                re.findall(
                    r"\b(best|optimal|efficient|clean|robust)\b", message.lower()
                )
            ),
        }

        return features

    async def _identify_user_goals(
        self, message: str, context: ConversationContext, entities: List[Entity]
    ) -> List[str]:
        """Identify user goals from the message"""

        goals = []

        # Goal patterns
        goal_patterns = [
            (r"i want to (.+)", "User wants to {}"),
            (r"i need (.+)", "User needs {}"),
            (r"help me (.+)", "Help user {}"),
            (r"can you (.+)", "User requests to {}"),
            (r"please (.+)", "User requests to {}"),
        ]

        for pattern, template in goal_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                goal = template.format(match.group(1))
                goals.append(goal)

        # If no explicit goals found, infer from intent and entities
        if not goals:
            if entities:
                entity_text = ", ".join([e.text for e in entities[:3]])
                goals.append(f"Work with {entity_text}")
            else:
                goals.append("Provide assistance with the request")

        return goals

    async def _identify_constraints(
        self, message: str, context: ConversationContext
    ) -> Dict[str, Any]:
        """Identify constraints from the message"""

        constraints = {}

        # Time constraints
        time_patterns = [
            (r"in (\d+) (minute|hour|day)s?", "time_limit"),
            (r"by (today|tomorrow|next week)", "deadline"),
            (r"(urgent|asap|quickly)", "urgency"),
        ]

        for pattern, constraint_type in time_patterns:
            if re.search(pattern, message.lower()):
                constraints[constraint_type] = True

        # Quality constraints
        quality_patterns = [
            (r"(simple|basic)", "complexity_preference"),
            (r"(detailed|comprehensive|thorough)", "detail_level"),
            (r"(production|enterprise|professional)", "quality_level"),
        ]

        for pattern, constraint_type in quality_patterns:
            if re.search(pattern, message.lower()):
                constraints[constraint_type] = pattern

        # Format constraints
        if any(
            fmt in message.lower() for fmt in ["pdf", "docx", "pptx", "json", "xml"]
        ):
            constraints["output_format"] = "specified"

        return constraints

    async def _classify_domain(
        self, message: str, entities: List[Entity], keywords: List[str]
    ) -> str:
        """Classify the domain of the query"""

        # Default domain
        domain = "software_engineering"

        # Check for specific domains based on entities and keywords
        domain_indicators = {
            "web_development": [
                "html",
                "css",
                "javascript",
                "react",
                "angular",
                "vue",
                "frontend",
                "backend",
            ],
            "data_science": [
                "python",
                "pandas",
                "numpy",
                "machine learning",
                "data analysis",
                "visualization",
            ],
            "mobile_development": [
                "android",
                "ios",
                "react native",
                "flutter",
                "mobile app",
            ],
            "devops": [
                "docker",
                "kubernetes",
                "jenkins",
                "ci/cd",
                "deployment",
                "infrastructure",
            ],
            "database": ["sql", "mongodb", "postgresql", "mysql", "database", "query"],
        }

        for domain_name, indicators in domain_indicators.items():
            if any(indicator in message.lower() for indicator in indicators):
                domain = domain_name
                break

        return domain

    async def _calculate_analysis_confidence(
        self, intent: IntentType, entities: List[Entity], complexity: ComplexityLevel
    ) -> float:
        """Calculate confidence score for the analysis"""

        confidence = 0.5  # Base confidence

        # Intent confidence
        if intent != IntentType.GENERAL_QUERY:
            confidence += 0.2

        # Entity confidence
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            confidence += avg_entity_confidence * 0.2

        # Complexity confidence
        if complexity != ComplexityLevel.SIMPLE:
            confidence += 0.1

        return min(confidence, 1.0)

    async def _identify_and_prioritize_goals(
        self, analysis: QueryAnalysis, context: ConversationContext
    ) -> List[Goal]:
        """Identify and prioritize user goals"""

        goals = []

        # Create goals based on analysis
        for i, goal_desc in enumerate(analysis.user_goals):
            goal = Goal(
                goal_id=f"goal_{i+1}",
                description=goal_desc,
                priority=self._calculate_goal_priority(goal_desc, analysis),
                success_criteria=self._define_success_criteria(goal_desc, analysis),
                estimated_effort=self._estimate_goal_effort(goal_desc, analysis),
            )
            goals.append(goal)

        # Sort by priority
        goals.sort(key=lambda g: g.priority, reverse=True)

        return goals

    def _calculate_goal_priority(self, goal_desc: str, analysis: QueryAnalysis) -> int:
        """Calculate priority for a goal"""

        priority = 1

        # Urgency indicators increase priority
        if analysis.constraints.get("urgency"):
            priority += 2

        # Complexity affects priority
        if analysis.complexity == ComplexityLevel.VERY_COMPLEX:
            priority += 1

        # Intent-based priority
        if analysis.intent in [
            IntentType.CODE_GENERATION,
            IntentType.DOCUMENT_GENERATION,
        ]:
            priority += 1

        return priority

    def _define_success_criteria(
        self, goal_desc: str, analysis: QueryAnalysis
    ) -> List[str]:
        """Define success criteria for a goal"""

        criteria = ["Task completed successfully"]

        # Add specific criteria based on intent
        if analysis.intent == IntentType.CODE_GENERATION:
            criteria.extend(
                [
                    "Code compiles without errors",
                    "Code follows best practices",
                    "Code includes appropriate documentation",
                ]
            )
        elif analysis.intent == IntentType.DOCUMENT_GENERATION:
            criteria.extend(
                [
                    "Document is properly formatted",
                    "Content is accurate and complete",
                    "Document is exported in requested format",
                ]
            )
        elif analysis.intent == IntentType.KNOWLEDGE_RETRIEVAL:
            criteria.extend(
                [
                    "Relevant information is retrieved",
                    "Information is accurate and up-to-date",
                    "Response addresses user's question",
                ]
            )

        return criteria

    def _estimate_goal_effort(self, goal_desc: str, analysis: QueryAnalysis) -> float:
        """Estimate effort required for a goal"""

        base_effort = 1.0

        # Complexity multiplier
        complexity_multipliers = {
            ComplexityLevel.SIMPLE: 1.0,
            ComplexityLevel.MODERATE: 1.5,
            ComplexityLevel.COMPLEX: 2.0,
            ComplexityLevel.VERY_COMPLEX: 3.0,
        }

        effort = base_effort * complexity_multipliers.get(analysis.complexity, 1.0)

        # Entity count affects effort
        effort += len(analysis.entities) * 0.2

        return effort

    async def _hierarchical_task_decomposition(
        self, goals: List[Goal], analysis: QueryAnalysis, context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """Hierarchical task decomposition based on goals and analysis"""

        tasks = []
        task_counter = 1

        for goal in goals:
            # Decompose each goal into tasks
            goal_tasks = await self._decompose_goal_into_tasks(
                goal, analysis, context, task_counter
            )
            tasks.extend(goal_tasks)
            task_counter += len(goal_tasks)

        return tasks

    async def _decompose_goal_into_tasks(
        self,
        goal: Goal,
        analysis: QueryAnalysis,
        context: ConversationContext,
        start_counter: int,
    ) -> List[Dict[str, Any]]:
        """Decompose a single goal into executable tasks"""

        tasks = []

        # Task decomposition based on intent
        if analysis.intent == IntentType.KNOWLEDGE_RETRIEVAL:
            tasks.extend(
                self._create_knowledge_retrieval_tasks(goal, analysis, start_counter)
            )

        elif analysis.intent == IntentType.CODE_GENERATION:
            tasks.extend(
                self._create_code_generation_tasks(goal, analysis, start_counter)
            )

        elif analysis.intent == IntentType.CONTENT_GENERATION:
            tasks.extend(
                self._create_content_generation_tasks(goal, analysis, start_counter)
            )

        elif analysis.intent == IntentType.DOCUMENT_GENERATION:
            tasks.extend(
                self._create_document_generation_tasks(goal, analysis, start_counter)
            )

        elif analysis.intent == IntentType.ANALYSIS:
            tasks.extend(self._create_analysis_tasks(goal, analysis, start_counter))

        elif analysis.intent == IntentType.MULTI_STEP:
            tasks.extend(self._create_multi_step_tasks(goal, analysis, start_counter))

        else:
            tasks.extend(self._create_general_tasks(goal, analysis, start_counter))

        return tasks

    def _create_knowledge_retrieval_tasks(
        self, goal: Goal, analysis: QueryAnalysis, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Create tasks for knowledge retrieval"""

        return [
            {
                "id": f"task_{start_counter}",
                "name": "retrieve_knowledge",
                "type": "tool_execution",
                "tool": "knowledge_retrieval",
                "parameters": {
                    "query": goal.description,
                    "entities": [e.text for e in analysis.entities],
                    "domain": analysis.domain,
                },
                "priority": goal.priority,
                "estimated_duration": 2.0,
                "goal_id": goal.goal_id,
            }
        ]

    def _create_code_generation_tasks(
        self, goal: Goal, analysis: QueryAnalysis, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Create tasks for code generation"""

        tasks = [
            {
                "id": f"task_{start_counter}",
                "name": "analyze_requirements",
                "type": "analysis",
                "parameters": {
                    "type": "code_requirements",
                    "goal": goal.description,
                    "entities": [e.text for e in analysis.entities],
                },
                "priority": goal.priority,
                "estimated_duration": 1.5,
                "goal_id": goal.goal_id,
            },
            {
                "id": f"task_{start_counter + 1}",
                "name": "generate_code",
                "type": "code_generation",
                "parameters": {
                    "language": self._detect_programming_language(analysis.entities),
                    "requirements_dependent": True,
                },
                "priority": goal.priority,
                "estimated_duration": 5.0,
                "goal_id": goal.goal_id,
            },
        ]

        # Add validation task if needed
        if analysis.complexity in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.VERY_COMPLEX,
        ]:
            tasks.append(
                {
                    "id": f"task_{start_counter + 2}",
                    "name": "validate_code",
                    "type": "tool_execution",
                    "tool": "compiler_runtime",
                    "parameters": {"validation_only": True},
                    "priority": goal.priority - 1,
                    "estimated_duration": 2.0,
                    "goal_id": goal.goal_id,
                }
            )

        return tasks

    def _create_content_generation_tasks(
        self, goal: Goal, analysis: QueryAnalysis, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Create tasks for content generation"""

        tasks = [
            {
                "id": f"task_{start_counter}",
                "name": "retrieve_context",
                "type": "tool_execution",
                "tool": "knowledge_retrieval",
                "parameters": {
                    "query": goal.description,
                    "context_for_generation": True,
                },
                "priority": goal.priority,
                "estimated_duration": 2.0,
                "goal_id": goal.goal_id,
            },
            {
                "id": f"task_{start_counter + 1}",
                "name": "generate_content",
                "type": "content_generation",
                "parameters": {
                    "context_dependent": True,
                    "complexity": analysis.complexity.value,
                    "constraints": analysis.constraints,
                },
                "priority": goal.priority,
                "estimated_duration": 4.0,
                "goal_id": goal.goal_id,
            },
        ]

        return tasks

    def _create_document_generation_tasks(
        self, goal: Goal, analysis: QueryAnalysis, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Create tasks for document generation"""

        tasks = [
            {
                "id": f"task_{start_counter}",
                "name": "prepare_content",
                "type": "content_generation",
                "parameters": {
                    "for_document": True,
                    "format": self._detect_document_format(analysis.entities),
                },
                "priority": goal.priority,
                "estimated_duration": 3.0,
                "goal_id": goal.goal_id,
            },
            {
                "id": f"task_{start_counter + 1}",
                "name": "generate_document",
                "type": "tool_execution",
                "tool": "document_generation",
                "parameters": {
                    "format": self._detect_document_format(analysis.entities),
                    "content_dependent": True,
                },
                "priority": goal.priority,
                "estimated_duration": 2.0,
                "goal_id": goal.goal_id,
            },
        ]

        return tasks

    def _create_analysis_tasks(
        self, goal: Goal, analysis: QueryAnalysis, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Create tasks for analysis"""

        return [
            {
                "id": f"task_{start_counter}",
                "name": "gather_data",
                "type": "tool_execution",
                "tool": "knowledge_retrieval",
                "parameters": {"query": goal.description, "analysis_context": True},
                "priority": goal.priority,
                "estimated_duration": 2.0,
                "goal_id": goal.goal_id,
            },
            {
                "id": f"task_{start_counter + 1}",
                "name": "perform_analysis",
                "type": "analysis",
                "parameters": {"type": "comprehensive", "data_dependent": True},
                "priority": goal.priority,
                "estimated_duration": 3.0,
                "goal_id": goal.goal_id,
            },
        ]

    def _create_multi_step_tasks(
        self, goal: Goal, analysis: QueryAnalysis, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Create tasks for multi-step requests"""

        # For multi-step, create a planning task first
        return [
            {
                "id": f"task_{start_counter}",
                "name": "decompose_multi_step",
                "type": "planning",
                "parameters": {
                    "multi_step_query": goal.description,
                    "entities": [e.text for e in analysis.entities],
                },
                "priority": goal.priority,
                "estimated_duration": 1.0,
                "goal_id": goal.goal_id,
            }
        ]

    def _create_general_tasks(
        self, goal: Goal, analysis: QueryAnalysis, start_counter: int
    ) -> List[Dict[str, Any]]:
        """Create tasks for general queries"""

        return [
            {
                "id": f"task_{start_counter}",
                "name": "general_response",
                "type": "general_processing",
                "parameters": {
                    "query": goal.description,
                    "context": analysis.semantic_features,
                },
                "priority": goal.priority,
                "estimated_duration": 1.0,
                "goal_id": goal.goal_id,
            }
        ]

    def _detect_programming_language(self, entities: List[Entity]) -> str:
        """Detect programming language from entities"""

        for entity in entities:
            if entity.entity_type == EntityType.PROGRAMMING_LANGUAGE:
                return entity.text

        return "auto_detect"

    def _detect_document_format(self, entities: List[Entity]) -> str:
        """Detect document format from entities"""

        for entity in entities:
            if entity.entity_type == EntityType.FILE_FORMAT:
                return entity.text

        return "docx"  # Default format

    async def _advanced_dependency_analysis(
        self, tasks: List[Dict[str, Any]], goals: List[Goal]
    ) -> Dict[str, List[str]]:
        """Advanced dependency analysis with goal relationships"""

        dependencies = {}

        # Initialize dependencies
        for task in tasks:
            dependencies[task["id"]] = []

        # Analyze dependencies within the same goal
        goal_tasks = {}
        for task in tasks:
            goal_id = task.get("goal_id", "default")
            if goal_id not in goal_tasks:
                goal_tasks[goal_id] = []
            goal_tasks[goal_id].append(task)

        # Add intra-goal dependencies
        for goal_id, goal_task_list in goal_tasks.items():
            sorted_tasks = sorted(
                goal_task_list, key=lambda x: x.get("priority", 1), reverse=True
            )

            for i in range(1, len(sorted_tasks)):
                current_task = sorted_tasks[i]
                prev_task = sorted_tasks[i - 1]

                # Add dependency if current task needs previous task's output
                if self._tasks_have_dependency(prev_task, current_task):
                    dependencies[current_task["id"]].append(prev_task["id"])

        # Add inter-goal dependencies
        for goal in goals:
            if goal.dependencies:
                goal_task_ids = [
                    t["id"] for t in tasks if t.get("goal_id") == goal.goal_id
                ]
                dep_task_ids = []

                for dep_goal_id in goal.dependencies:
                    dep_task_ids.extend(
                        [t["id"] for t in tasks if t.get("goal_id") == dep_goal_id]
                    )

                # Make all tasks in current goal depend on all tasks in dependency goals
                for task_id in goal_task_ids:
                    dependencies[task_id].extend(dep_task_ids)

        return dependencies

    def _tasks_have_dependency(
        self, task1: Dict[str, Any], task2: Dict[str, Any]
    ) -> bool:
        """Check if task2 depends on task1"""

        # Content generation depends on context retrieval
        if (
            task1.get("name") == "retrieve_context"
            and task2.get("name") == "generate_content"
        ):
            return True

        # Code validation depends on code generation
        if (
            task1.get("name") == "generate_code"
            and task2.get("name") == "validate_code"
        ):
            return True

        # Document generation depends on content preparation
        if (
            task1.get("name") == "prepare_content"
            and task2.get("name") == "generate_document"
        ):
            return True

        # Analysis depends on data gathering
        if (
            task1.get("name") == "gather_data"
            and task2.get("name") == "perform_analysis"
        ):
            return True

        # General dependency based on parameters
        if task2.get("parameters", {}).get("context_dependent") or task2.get(
            "parameters", {}
        ).get("data_dependent"):
            return True

        return False

    async def _estimate_duration_advanced(
        self, tasks: List[Dict[str, Any]], dependencies: Dict[str, List[str]]
    ) -> float:
        """Advanced duration estimation considering dependencies and parallelization"""

        # Calculate critical path
        task_durations = {
            task["id"]: task.get("estimated_duration", 1.0) for task in tasks
        }

        # Simple critical path calculation
        def calculate_earliest_start(
            task_id: str, memo: Dict[str, float] = None
        ) -> float:
            if memo is None:
                memo = {}

            if task_id in memo:
                return memo[task_id]

            deps = dependencies.get(task_id, [])
            if not deps:
                memo[task_id] = 0.0
                return 0.0

            max_dep_finish = max(
                calculate_earliest_start(dep_id, memo) + task_durations[dep_id]
                for dep_id in deps
            )

            memo[task_id] = max_dep_finish
            return max_dep_finish

        # Calculate total duration (critical path)
        max_finish_time = 0.0
        for task in tasks:
            task_id = task["id"]
            earliest_start = calculate_earliest_start(task_id)
            finish_time = earliest_start + task_durations[task_id]
            max_finish_time = max(max_finish_time, finish_time)

        return max_finish_time

    async def _plan_recovery_strategies(
        self, tasks: List[Dict[str, Any]], analysis: QueryAnalysis
    ) -> Dict[str, Dict[str, Any]]:
        """Plan recovery strategies for tasks"""

        recovery_strategies = {}

        for task in tasks:
            task_id = task["id"]
            task_type = task.get("type", "general")

            if task_type == "tool_execution":
                recovery_strategies[task_id] = {
                    "strategy": "retry_with_fallback",
                    "max_retries": 3,
                    "fallback_tool": self._get_fallback_tool(task.get("tool")),
                    "timeout": 300,
                }
            elif task_type == "code_generation":
                recovery_strategies[task_id] = {
                    "strategy": "simplify_and_retry",
                    "max_retries": 2,
                    "fallback_approach": "basic_implementation",
                    "timeout": 600,
                }
            elif task_type == "content_generation":
                recovery_strategies[task_id] = {
                    "strategy": "reduce_complexity",
                    "max_retries": 2,
                    "fallback_approach": "template_based",
                    "timeout": 300,
                }
            else:
                recovery_strategies[task_id] = {
                    "strategy": "retry",
                    "max_retries": 2,
                    "timeout": 180,
                }

        return recovery_strategies

    def _get_fallback_tool(self, primary_tool: str) -> Optional[str]:
        """Get fallback tool for a primary tool"""

        fallback_map = {
            "knowledge_retrieval": "simple_search",
            "document_generation": "text_generation",
            "compiler_runtime": "syntax_checker",
        }

        return fallback_map.get(primary_tool)

    async def _identify_parallel_execution_groups(
        self, tasks: List[Dict[str, Any]], dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Identify groups of tasks that can be executed in parallel"""

        parallel_groups = []
        processed_tasks = set()

        # Find tasks with no dependencies (can start immediately)
        independent_tasks = [
            task["id"] for task in tasks if not dependencies.get(task["id"], [])
        ]

        if len(independent_tasks) > 1:
            parallel_groups.append(independent_tasks)

        processed_tasks.update(independent_tasks)

        # Find subsequent parallel groups
        while len(processed_tasks) < len(tasks):
            next_parallel_group = []

            for task in tasks:
                task_id = task["id"]
                if task_id in processed_tasks:
                    continue

                # Check if all dependencies are satisfied
                task_deps = dependencies.get(task_id, [])
                if all(dep in processed_tasks for dep in task_deps):
                    next_parallel_group.append(task_id)

            if next_parallel_group:
                if len(next_parallel_group) > 1:
                    parallel_groups.append(next_parallel_group)
                processed_tasks.update(next_parallel_group)
            else:
                # No more tasks can be processed, break to avoid infinite loop
                break

        return parallel_groups

    def _determine_priority_advanced(
        self, analysis: QueryAnalysis, goals: List[Goal]
    ) -> int:
        """Advanced priority determination"""

        base_priority = 1

        # Complexity-based priority
        complexity_priority = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 2,
            ComplexityLevel.COMPLEX: 3,
            ComplexityLevel.VERY_COMPLEX: 4,
        }

        priority = complexity_priority.get(analysis.complexity, 1)

        # Urgency boost
        if analysis.constraints.get("urgency"):
            priority += 2

        # Goal priority influence
        if goals:
            max_goal_priority = max(goal.priority for goal in goals)
            priority = max(priority, max_goal_priority)

        return priority

    async def adapt_plan(
        self,
        plan: ExecutionPlan,
        feedback: Dict[str, Any],
        context: ConversationContext,
    ) -> ExecutionPlan:
        """Adapt existing plan based on feedback"""

        logger.info("Adapting execution plan", plan_id=plan.plan_id)

        # Analyze feedback
        feedback_type = feedback.get("type", "general")

        if feedback_type == "task_failed":
            # Handle task failure
            failed_task_id = feedback.get("task_id")
            plan = await self._handle_task_failure(plan, failed_task_id, feedback)

        elif feedback_type == "user_clarification":
            # Handle user clarification
            plan = await self._incorporate_user_clarification(plan, feedback, context)

        elif feedback_type == "resource_constraint":
            # Handle resource constraints
            plan = await self._adapt_for_resource_constraints(plan, feedback)

        elif feedback_type == "priority_change":
            # Handle priority changes
            plan = await self._adapt_for_priority_change(plan, feedback)

        logger.info("Plan adaptation completed", plan_id=plan.plan_id)
        return plan

    async def _handle_task_failure(
        self, plan: ExecutionPlan, failed_task_id: str, feedback: Dict[str, Any]
    ) -> ExecutionPlan:
        """Handle task failure and adapt plan"""

        # Find the failed task
        failed_task = None
        for task in plan.tasks:
            if task["id"] == failed_task_id:
                failed_task = task
                break

        if not failed_task:
            return plan

        # Apply recovery strategy
        recovery_strategy = plan.recovery_strategies.get(failed_task_id, {})
        strategy_type = recovery_strategy.get("strategy", "retry")

        if strategy_type == "retry_with_fallback":
            # Modify task to use fallback tool
            fallback_tool = recovery_strategy.get("fallback_tool")
            if fallback_tool:
                failed_task["tool"] = fallback_tool
                failed_task["parameters"]["fallback_mode"] = True

        elif strategy_type == "simplify_and_retry":
            # Simplify task parameters
            failed_task["parameters"]["complexity"] = "simple"
            failed_task["parameters"]["fallback_mode"] = True

        return plan

    async def _incorporate_user_clarification(
        self,
        plan: ExecutionPlan,
        feedback: Dict[str, Any],
        context: ConversationContext,
    ) -> ExecutionPlan:
        """Incorporate user clarification into the plan"""

        clarification = feedback.get("clarification", "")

        # Re-analyze with clarification
        combined_message = f"{context.message_history[-1]['content']} {clarification}"
        updated_context = ConversationContext(
            session_id=context.session_id,
            user_id=context.user_id,
            message_history=context.message_history
            + [{"content": clarification, "role": "user"}],
            user_preferences=context.user_preferences,
            current_topic=context.current_topic,
        )

        # Create new plan with clarification
        new_plan = await self.create_plan(combined_message, updated_context)

        # Merge with existing plan
        plan.tasks.extend(new_plan.tasks)
        plan.dependencies.update(new_plan.dependencies)
        plan.estimated_duration += new_plan.estimated_duration

        return plan

    async def _adapt_for_resource_constraints(
        self, plan: ExecutionPlan, feedback: Dict[str, Any]
    ) -> ExecutionPlan:
        """Adapt plan for resource constraints"""

        constraint_type = feedback.get("constraint_type", "time")

        if constraint_type == "time":
            # Reduce estimated durations and simplify tasks
            for task in plan.tasks:
                task["estimated_duration"] *= 0.8
                if "parameters" in task:
                    task["parameters"]["time_constrained"] = True

        elif constraint_type == "complexity":
            # Simplify complex tasks
            for task in plan.tasks:
                if task.get("type") == "code_generation":
                    task["parameters"]["complexity"] = "simple"
                elif task.get("type") == "content_generation":
                    task["parameters"]["detail_level"] = "basic"

        return plan

    async def _adapt_for_priority_change(
        self, plan: ExecutionPlan, feedback: Dict[str, Any]
    ) -> ExecutionPlan:
        """Adapt plan for priority changes"""

        new_priority = feedback.get("new_priority", plan.priority)
        plan.priority = new_priority

        # Update task priorities
        for task in plan.tasks:
            task["priority"] = new_priority

        return plan

    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize intent classification patterns"""
        return {
            "knowledge_retrieval": [
                "explain",
                "what is",
                "how does",
                "why",
                "tell me about",
                "describe",
            ],
            "code_generation": [
                "write code",
                "implement",
                "create function",
                "build class",
                "develop",
                "program",
            ],
            "content_generation": [
                "generate",
                "create content",
                "write",
                "compose",
                "draft",
            ],
            "document_generation": [
                "generate document",
                "create report",
                "write documentation",
                "export to",
                "save as",
            ],
            "analysis": [
                "analyze",
                "review",
                "evaluate",
                "assess",
                "compare",
                "examine",
            ],
            "multi_step": [
                "and then",
                "after that",
                "next",
                "also",
                "additionally",
                "furthermore",
            ],
        }

    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns"""
        return {
            "programming_languages": [
                "python",
                "javascript",
                "java",
                "c++",
                "c#",
                "go",
                "rust",
                "typescript",
            ],
            "frameworks": [
                "react",
                "angular",
                "vue",
                "django",
                "flask",
                "spring",
                "express",
                "fastapi",
            ],
            "file_formats": [
                "pdf",
                "docx",
                "pptx",
                "json",
                "xml",
                "csv",
                "yaml",
                "markdown",
            ],
            "tools": [
                "git",
                "docker",
                "kubernetes",
                "jenkins",
                "github",
                "gitlab",
                "aws",
                "azure",
            ],
        }

    def _initialize_task_templates(self) -> Dict[str, TaskTemplate]:
        """Initialize task templates"""
        return {
            "knowledge_retrieval": TaskTemplate(
                task_type="tool_execution",
                required_inputs=["query"],
                optional_inputs=["filters", "max_results"],
                expected_outputs=["retrieved_content"],
                estimated_duration=2.0,
                resource_requirements={"memory": "low", "cpu": "medium"},
                failure_modes=["no_results", "timeout", "service_unavailable"],
                recovery_strategies=["retry", "broaden_query", "fallback_search"],
            ),
            "code_generation": TaskTemplate(
                task_type="code_generation",
                required_inputs=["requirements"],
                optional_inputs=["language", "framework", "style"],
                expected_outputs=["generated_code"],
                estimated_duration=5.0,
                resource_requirements={"memory": "medium", "cpu": "high"},
                failure_modes=["syntax_error", "logic_error", "timeout"],
                recovery_strategies=["simplify", "retry", "template_based"],
            ),
        }

    def _initialize_complexity_indicators(self) -> Dict[str, int]:
        """Initialize complexity indicators"""
        return {
            "multi_step": 2,
            "integration": 2,
            "optimization": 1,
            "scalability": 1,
            "security": 1,
            "performance": 1,
        }

    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords"""
        return {
            "web_development": [
                "html",
                "css",
                "javascript",
                "react",
                "angular",
                "vue",
                "frontend",
                "backend",
            ],
            "data_science": [
                "python",
                "pandas",
                "numpy",
                "machine learning",
                "data analysis",
                "visualization",
            ],
            "mobile_development": [
                "android",
                "ios",
                "react native",
                "flutter",
                "mobile app",
            ],
            "devops": [
                "docker",
                "kubernetes",
                "jenkins",
                "ci/cd",
                "deployment",
                "infrastructure",
            ],
            "database": ["sql", "mongodb", "postgresql", "mysql", "database", "query"],
        }

    def _initialize_planning_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize planning strategies"""
        return {
            "simple": {"max_tasks": 3, "max_depth": 2, "parallelization": False},
            "moderate": {"max_tasks": 7, "max_depth": 3, "parallelization": True},
            "complex": {"max_tasks": 15, "max_depth": 4, "parallelization": True},
            "very_complex": {"max_tasks": 25, "max_depth": 5, "parallelization": True},
        }

    async def _initialize_goal_decomposition(self):
        """Initialize goal decomposition components"""
        logger.info("Goal decomposition system initialized")

    async def _initialize_dependency_analyzer(self):
        """Initialize dependency analysis components"""
        logger.info("Dependency analyzer initialized")

    async def _initialize_adaptation_engine(self):
        """Initialize plan adaptation engine"""
        logger.info("Plan adaptation engine initialized")

    async def shutdown(self):
        """Shutdown planning module"""
        logger.info("Shutting down Planning Module")
        # Cleanup resources if needed
