"""
Unit tests for agent server planning module
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from src.agent_server.planning import (
    PlanningModule,
    IntentType,
    ComplexityLevel,
    EntityType,
    ConversationContext,
    Entity,
    QueryAnalysis,
    Goal,
)


@pytest.mark.unit
class TestPlanningModule:
    """Test PlanningModule class."""

    def test_planning_module_initialization(self):
        """Test PlanningModule initialization."""
        module = PlanningModule()

        assert module._initialized is False
        assert module.intent_patterns is not None
        assert module.entity_patterns is not None
        assert module.task_templates is not None

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test planning module initialization."""
        module = PlanningModule()

        with patch.object(
            module, "_initialize_goal_decomposition", new_callable=AsyncMock
        ):
            with patch.object(
                module, "_initialize_dependency_analyzer", new_callable=AsyncMock
            ):
                with patch.object(
                    module, "_initialize_adaptation_engine", new_callable=AsyncMock
                ):
                    await module.initialize()

        assert module._initialized is True

    @pytest.mark.asyncio
    async def test_create_plan(self):
        """Test plan creation."""
        module = PlanningModule()
        module._initialized = True

        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        with patch.object(module, "_analyze_query_comprehensive") as mock_analyze:
            with patch.object(module, "_identify_and_prioritize_goals") as mock_goals:
                with patch.object(
                    module, "_hierarchical_task_decomposition"
                ) as mock_tasks:
                    with patch.object(
                        module, "_advanced_dependency_analysis"
                    ) as mock_deps:
                        with patch.object(
                            module, "_estimate_duration_advanced"
                        ) as mock_duration:
                            with patch.object(
                                module, "_plan_recovery_strategies"
                            ) as mock_recovery:
                                with patch.object(
                                    module, "_identify_parallel_execution_groups"
                                ) as mock_parallel:

                                    # Setup mocks
                                    mock_analyze.return_value = QueryAnalysis(
                                        intent=IntentType.CODE_GENERATION,
                                        entities=[],
                                        complexity=ComplexityLevel.MODERATE,
                                        domain="programming",
                                        confidence=0.8,
                                    )
                                    mock_goals.return_value = [
                                        Goal(
                                            goal_id="goal_1",
                                            description="Generate Python code",
                                            priority=1,
                                            success_criteria=["Code compiles"],
                                        )
                                    ]
                                    mock_tasks.return_value = [
                                        {
                                            "id": "task_1",
                                            "name": "analyze_requirements",
                                            "type": "analysis",
                                        }
                                    ]
                                    mock_deps.return_value = {"task_1": []}
                                    mock_duration.return_value = 5.0
                                    mock_recovery.return_value = {}
                                    mock_parallel.return_value = []

                                    plan = await module.create_plan(
                                        "Write a Python function", context
                                    )

        assert plan is not None
        assert plan.plan_id is not None
        assert len(plan.tasks) == 1
        assert plan.estimated_duration == 5.0


@pytest.mark.unit
class TestQueryAnalysis:
    """Test query analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_query_comprehensive(self):
        """Test comprehensive query analysis."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        with patch.object(module, "_classify_intent_advanced") as mock_intent:
            with patch.object(module, "_extract_entities_advanced") as mock_entities:
                with patch.object(
                    module, "_assess_complexity_advanced"
                ) as mock_complexity:
                    with patch.object(module, "_extract_keywords") as mock_keywords:
                        with patch.object(
                            module, "_extract_semantic_features"
                        ) as mock_features:
                            with patch.object(
                                module, "_identify_user_goals"
                            ) as mock_goals:
                                with patch.object(
                                    module, "_identify_constraints"
                                ) as mock_constraints:
                                    with patch.object(
                                        module, "_classify_domain"
                                    ) as mock_domain:
                                        with patch.object(
                                            module, "_calculate_analysis_confidence"
                                        ) as mock_confidence:

                                            # Setup mocks
                                            mock_intent.return_value = (
                                                IntentType.CODE_GENERATION
                                            )
                                            mock_entities.return_value = [
                                                Entity(
                                                    text="python",
                                                    entity_type=EntityType.PROGRAMMING_LANGUAGE,
                                                    confidence=0.9,
                                                )
                                            ]
                                            mock_complexity.return_value = (
                                                ComplexityLevel.MODERATE
                                            )
                                            mock_keywords.return_value = [
                                                "python",
                                                "function",
                                            ]
                                            mock_features.return_value = {
                                                "imperative_verbs": 1
                                            }
                                            mock_goals.return_value = [
                                                "Create Python function"
                                            ]
                                            mock_constraints.return_value = {}
                                            mock_domain.return_value = "programming"
                                            mock_confidence.return_value = 0.85

                                            analysis = await module._analyze_query_comprehensive(
                                                "Write a Python function", context
                                            )

        assert analysis.intent == IntentType.CODE_GENERATION
        assert len(analysis.entities) == 1
        assert analysis.complexity == ComplexityLevel.MODERATE
        assert analysis.confidence == 0.85

    @pytest.mark.asyncio
    async def test_classify_intent_advanced_code_generation(self):
        """Test intent classification for code generation."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        intent = await module._classify_intent_advanced(
            "Write a Python function", context
        )
        assert intent == IntentType.CODE_GENERATION

    @pytest.mark.asyncio
    async def test_classify_intent_advanced_knowledge_retrieval(self):
        """Test intent classification for knowledge retrieval."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        intent = await module._classify_intent_advanced("What is Python?", context)
        assert intent == IntentType.KNOWLEDGE_RETRIEVAL

    @pytest.mark.asyncio
    async def test_classify_intent_advanced_multi_step(self):
        """Test intent classification for multi-step queries."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        intent = await module._classify_intent_advanced(
            "First explain Python, then write a function", context
        )
        assert intent == IntentType.MULTI_STEP

    @pytest.mark.asyncio
    async def test_extract_entities_advanced(self):
        """Test advanced entity extraction."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        entities = await module._extract_entities_advanced(
            "Write a Python function using Django framework", context
        )

        # Should extract both Python and Django
        entity_texts = [e.text for e in entities]
        assert "python" in entity_texts
        assert "django" in entity_texts

        # Check entity types
        python_entity = next(e for e in entities if e.text == "python")
        django_entity = next(e for e in entities if e.text == "django")

        assert python_entity.entity_type == EntityType.PROGRAMMING_LANGUAGE
        assert django_entity.entity_type == EntityType.FRAMEWORK

    @pytest.mark.asyncio
    async def test_assess_complexity_advanced_simple(self):
        """Test complexity assessment for simple queries."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        complexity = await module._assess_complexity_advanced(
            "What is Python?", context, []
        )

        assert complexity == ComplexityLevel.SIMPLE

    @pytest.mark.asyncio
    async def test_assess_complexity_advanced_complex(self):
        """Test complexity assessment for complex queries."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        entities = [
            Entity("python", EntityType.PROGRAMMING_LANGUAGE, 0.9),
            Entity("django", EntityType.FRAMEWORK, 0.8),
            Entity("api", EntityType.CONCEPT, 0.7),
        ]

        complexity = await module._assess_complexity_advanced(
            "Create a comprehensive Django REST API with authentication, "
            "database integration, and deployment configuration. "
            "Also include testing and documentation.",
            context,
            entities,
        )

        assert complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]

    @pytest.mark.asyncio
    async def test_extract_keywords(self):
        """Test keyword extraction."""
        module = PlanningModule()

        keywords = await module._extract_keywords(
            "Create a Python function for data processing and analysis"
        )

        assert "python" in keywords
        assert "function" in keywords
        assert "data" in keywords
        assert "processing" in keywords
        assert "analysis" in keywords

        # Should not include stop words
        assert "a" not in keywords
        assert "for" not in keywords

    @pytest.mark.asyncio
    async def test_extract_semantic_features(self):
        """Test semantic feature extraction."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        features = await module._extract_semantic_features(
            "What is the best way to create a Python API?", context
        )

        assert features["question_words"] > 0  # "what"
        assert features["imperative_verbs"] > 0  # "create"
        assert features["technical_terms"] > 0  # "api"
        assert features["quality_indicators"] > 0  # "best"


@pytest.mark.unit
class TestGoalManagement:
    """Test goal identification and management."""

    @pytest.mark.asyncio
    async def test_identify_and_prioritize_goals(self):
        """Test goal identification and prioritization."""
        module = PlanningModule()

        analysis = QueryAnalysis(
            intent=IntentType.CODE_GENERATION,
            entities=[Entity("python", EntityType.PROGRAMMING_LANGUAGE, 0.9)],
            complexity=ComplexityLevel.MODERATE,
            domain="programming",
            confidence=0.8,
            user_goals=["Create Python function", "Add documentation"],
        )

        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        goals = await module._identify_and_prioritize_goals(analysis, context)

        assert len(goals) == 2
        assert all(isinstance(goal, Goal) for goal in goals)
        assert goals[0].priority >= goals[1].priority  # Should be sorted by priority

    def test_calculate_goal_priority(self):
        """Test goal priority calculation."""
        module = PlanningModule()

        analysis = QueryAnalysis(
            intent=IntentType.CODE_GENERATION,
            entities=[],
            complexity=ComplexityLevel.VERY_COMPLEX,
            domain="programming",
            confidence=0.8,
            constraints={"urgency": True},
        )

        priority = module._calculate_goal_priority("Create urgent feature", analysis)

        assert priority > 1  # Should have higher priority due to urgency and complexity

    def test_define_success_criteria(self):
        """Test success criteria definition."""
        module = PlanningModule()

        analysis = QueryAnalysis(
            intent=IntentType.CODE_GENERATION,
            entities=[],
            complexity=ComplexityLevel.MODERATE,
            domain="programming",
            confidence=0.8,
        )

        criteria = module._define_success_criteria("Create Python function", analysis)

        assert "Task completed successfully" in criteria
        assert "Code compiles without errors" in criteria
        assert "Code follows best practices" in criteria

    def test_estimate_goal_effort(self):
        """Test goal effort estimation."""
        module = PlanningModule()

        analysis = QueryAnalysis(
            intent=IntentType.CODE_GENERATION,
            entities=[Entity("python", EntityType.PROGRAMMING_LANGUAGE, 0.9)],
            complexity=ComplexityLevel.COMPLEX,
            domain="programming",
            confidence=0.8,
        )

        effort = module._estimate_goal_effort("Create complex system", analysis)

        assert effort > 1.0  # Should be higher than base effort due to complexity


@pytest.mark.unit
class TestTaskDecomposition:
    """Test task decomposition functionality."""

    @pytest.mark.asyncio
    async def test_hierarchical_task_decomposition(self):
        """Test hierarchical task decomposition."""
        module = PlanningModule()

        goals = [
            Goal(
                goal_id="goal_1",
                description="Create Python function",
                priority=1,
                success_criteria=["Code compiles"],
            )
        ]

        analysis = QueryAnalysis(
            intent=IntentType.CODE_GENERATION,
            entities=[Entity("python", EntityType.PROGRAMMING_LANGUAGE, 0.9)],
            complexity=ComplexityLevel.MODERATE,
            domain="programming",
            confidence=0.8,
        )

        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        tasks = await module._hierarchical_task_decomposition(goals, analysis, context)

        assert len(tasks) > 0
        assert all("id" in task for task in tasks)
        assert all("name" in task for task in tasks)
        assert all("type" in task for task in tasks)

    def test_create_knowledge_retrieval_tasks(self):
        """Test knowledge retrieval task creation."""
        module = PlanningModule()

        goal = Goal(
            goal_id="goal_1",
            description="Learn about Python",
            priority=1,
            success_criteria=["Information retrieved"],
        )

        analysis = QueryAnalysis(
            intent=IntentType.KNOWLEDGE_RETRIEVAL,
            entities=[Entity("python", EntityType.PROGRAMMING_LANGUAGE, 0.9)],
            complexity=ComplexityLevel.SIMPLE,
            domain="programming",
            confidence=0.8,
        )

        tasks = module._create_knowledge_retrieval_tasks(goal, analysis, 1)

        assert len(tasks) == 1
        assert tasks[0]["type"] == "tool_execution"
        assert tasks[0]["tool"] == "knowledge_retrieval"
        assert "python" in tasks[0]["parameters"]["entities"]

    def test_create_code_generation_tasks(self):
        """Test code generation task creation."""
        module = PlanningModule()

        goal = Goal(
            goal_id="goal_1",
            description="Create Python function",
            priority=1,
            success_criteria=["Code compiles"],
        )

        analysis = QueryAnalysis(
            intent=IntentType.CODE_GENERATION,
            entities=[Entity("python", EntityType.PROGRAMMING_LANGUAGE, 0.9)],
            complexity=ComplexityLevel.MODERATE,
            domain="programming",
            confidence=0.8,
        )

        tasks = module._create_code_generation_tasks(goal, analysis, 1)

        assert len(tasks) >= 2  # Should have analysis and generation tasks
        assert any(task["name"] == "analyze_requirements" for task in tasks)
        assert any(task["name"] == "generate_code" for task in tasks)

    def test_detect_programming_language(self):
        """Test programming language detection."""
        module = PlanningModule()

        entities = [
            Entity("python", EntityType.PROGRAMMING_LANGUAGE, 0.9),
            Entity("django", EntityType.FRAMEWORK, 0.8),
        ]

        language = module._detect_programming_language(entities)
        assert language == "python"

        # Test with no language entities
        entities_no_lang = [Entity("api", EntityType.CONCEPT, 0.7)]
        language_default = module._detect_programming_language(entities_no_lang)
        assert language_default == "python"  # Should default to Python


@pytest.mark.unit
class TestPlanningEdgeCases:
    """Test planning module edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_query_analysis(self):
        """Test analysis of empty query."""
        module = PlanningModule()
        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        with patch.object(module, "_classify_intent_advanced") as mock_intent:
            with patch.object(module, "_extract_entities_advanced") as mock_entities:
                with patch.object(
                    module, "_assess_complexity_advanced"
                ) as mock_complexity:
                    with patch.object(module, "_extract_keywords") as mock_keywords:
                        with patch.object(
                            module, "_extract_semantic_features"
                        ) as mock_features:
                            with patch.object(
                                module, "_identify_user_goals"
                            ) as mock_goals:
                                with patch.object(
                                    module, "_identify_constraints"
                                ) as mock_constraints:
                                    with patch.object(
                                        module, "_classify_domain"
                                    ) as mock_domain:
                                        with patch.object(
                                            module, "_calculate_analysis_confidence"
                                        ) as mock_confidence:

                                            # Setup mocks for empty query
                                            mock_intent.return_value = (
                                                IntentType.GENERAL_QUERY
                                            )
                                            mock_entities.return_value = []
                                            mock_complexity.return_value = (
                                                ComplexityLevel.SIMPLE
                                            )
                                            mock_keywords.return_value = []
                                            mock_features.return_value = {}
                                            mock_goals.return_value = [
                                                "Provide assistance"
                                            ]
                                            mock_constraints.return_value = {}
                                            mock_domain.return_value = "general"
                                            mock_confidence.return_value = 0.1

                                            analysis = await module._analyze_query_comprehensive(
                                                "", context
                                            )

        assert analysis.intent == IntentType.GENERAL_QUERY
        assert len(analysis.entities) == 0
        assert analysis.complexity == ComplexityLevel.SIMPLE
        assert analysis.confidence == 0.1

    @pytest.mark.asyncio
    async def test_plan_creation_with_no_goals(self):
        """Test plan creation when no goals are identified."""
        module = PlanningModule()
        module._initialized = True

        context = ConversationContext(
            session_id="test_session",
            user_id="test_user",
            message_history=[],
            user_preferences={},
        )

        with patch.object(module, "_analyze_query_comprehensive") as mock_analyze:
            with patch.object(module, "_identify_and_prioritize_goals") as mock_goals:
                with patch.object(
                    module, "_hierarchical_task_decomposition"
                ) as mock_tasks:
                    with patch.object(
                        module, "_advanced_dependency_analysis"
                    ) as mock_deps:
                        with patch.object(
                            module, "_estimate_duration_advanced"
                        ) as mock_duration:
                            with patch.object(
                                module, "_plan_recovery_strategies"
                            ) as mock_recovery:
                                with patch.object(
                                    module, "_identify_parallel_execution_groups"
                                ) as mock_parallel:

                                    # Setup mocks for no goals scenario
                                    mock_analyze.return_value = QueryAnalysis(
                                        intent=IntentType.GENERAL_QUERY,
                                        entities=[],
                                        complexity=ComplexityLevel.SIMPLE,
                                        domain="general",
                                        confidence=0.1,
                                    )
                                    mock_goals.return_value = []  # No goals
                                    mock_tasks.return_value = []
                                    mock_deps.return_value = {}
                                    mock_duration.return_value = 0.0
                                    mock_recovery.return_value = {}
                                    mock_parallel.return_value = []

                                    plan = await module.create_plan("", context)

        assert plan is not None
        assert len(plan.tasks) == 0
        assert plan.estimated_duration == 0.0
