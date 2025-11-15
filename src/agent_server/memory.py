"""
Enhanced Memory Management
Advanced conversation context and user preference management with intelligent context pruning
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
import redis.asyncio as redis
from collections import defaultdict, deque

from .planning import ConversationContext
from .orchestrator import ExecutionResult
from src.shared.logging import get_logger
from src.shared.config import get_settings

logger = get_logger(__name__)


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContextRelevance(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


class MemoryType(Enum):
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class Message:
    content: str
    role: MessageRole
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 1.0
    context_tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    message_id: str = field(default_factory=lambda: str(datetime.utcnow().timestamp()))


@dataclass
class ConversationSummary:
    session_id: str
    summary_text: str
    key_topics: List[str]
    important_decisions: List[str]
    user_goals: List[str]
    created_at: datetime
    covers_messages: int
    relevance_score: float = 1.0


@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    expertise_areas: List[str] = field(default_factory=list)
    learning_style: str = "adaptive"
    communication_style: str = "balanced"
    frequent_topics: Dict[str, int] = field(default_factory=dict)
    tool_preferences: Dict[str, float] = field(default_factory=dict)
    success_patterns: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContextWindow:
    messages: List[Message]
    summary: Optional[ConversationSummary]
    relevant_history: List[Message]
    user_context: Dict[str, Any]
    session_state: Dict[str, Any]
    total_relevance_score: float
    window_size: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserPreferences:
    preferred_response_length: str = "medium"  # short, medium, long, adaptive
    preferred_complexity: str = "adaptive"  # simple, moderate, complex, adaptive
    preferred_formats: List[str] = field(default_factory=lambda: ["markdown"])
    notification_settings: Dict[str, bool] = field(
        default_factory=lambda: {"email": False, "push": True}
    )
    language_preference: str = "en"
    timezone: str = "UTC"
    accessibility_needs: List[str] = field(default_factory=list)
    interaction_style: str = "conversational"  # formal, conversational, technical
    explanation_depth: str = "balanced"  # brief, balanced, detailed
    code_style_preferences: Dict[str, str] = field(default_factory=dict)
    tool_preferences: Dict[str, float] = field(default_factory=dict)


class ContextPruner:
    """Intelligent context pruning based on relevance and recency"""

    def __init__(self):
        self.relevance_weights = {
            "recency": 0.3,
            "topic_similarity": 0.25,
            "user_engagement": 0.2,
            "task_relevance": 0.15,
            "semantic_similarity": 0.1,
        }

    async def prune_context(
        self,
        messages: List[Message],
        current_query: str,
        target_size: int,
        user_profile: Optional[UserProfile] = None,
    ) -> List[Message]:
        """Intelligently prune context to target size while preserving relevance"""

        if len(messages) <= target_size:
            return messages

        # Calculate relevance scores for all messages
        scored_messages = []
        for msg in messages:
            relevance = await self._calculate_message_relevance(
                msg, current_query, messages, user_profile
            )
            scored_messages.append((msg, relevance))

        # Sort by relevance (descending)
        scored_messages.sort(key=lambda x: x[1], reverse=True)

        # Select top messages while maintaining conversation flow
        selected_messages = await self._maintain_conversation_flow(
            scored_messages[: target_size * 2], target_size
        )

        # Sort selected messages by timestamp to maintain chronological order
        selected_messages.sort(key=lambda x: x.timestamp)

        logger.info(
            "Context pruned",
            original_size=len(messages),
            pruned_size=len(selected_messages),
            target_size=target_size,
        )

        return selected_messages

    async def _calculate_message_relevance(
        self,
        message: Message,
        current_query: str,
        all_messages: List[Message],
        user_profile: Optional[UserProfile],
    ) -> float:
        """Calculate relevance score for a message"""

        relevance_score = 0.0

        # Recency score (more recent = higher score)
        time_diff = (datetime.utcnow() - message.timestamp).total_seconds()
        max_time = 3600 * 24  # 24 hours
        recency_score = max(0, 1 - (time_diff / max_time))
        relevance_score += recency_score * self.relevance_weights["recency"]

        # Topic similarity score
        topic_score = await self._calculate_topic_similarity(message, current_query)
        relevance_score += topic_score * self.relevance_weights["topic_similarity"]

        # User engagement score (based on message length and role)
        engagement_score = self._calculate_engagement_score(message)
        relevance_score += engagement_score * self.relevance_weights["user_engagement"]

        # Task relevance score
        task_score = await self._calculate_task_relevance(message, current_query)
        relevance_score += task_score * self.relevance_weights["task_relevance"]

        # Semantic similarity score (if embeddings available)
        if message.embedding:
            semantic_score = await self._calculate_semantic_similarity(
                message, current_query
            )
            relevance_score += (
                semantic_score * self.relevance_weights["semantic_similarity"]
            )

        return min(relevance_score, 1.0)

    async def _calculate_topic_similarity(
        self, message: Message, current_query: str
    ) -> float:
        """Calculate topic similarity between message and current query"""

        # Simple keyword-based similarity (can be enhanced with NLP)
        message_words = set(message.content.lower().split())
        query_words = set(current_query.lower().split())

        if not message_words or not query_words:
            return 0.0

        intersection = message_words.intersection(query_words)
        union = message_words.union(query_words)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_engagement_score(self, message: Message) -> float:
        """Calculate user engagement score for a message"""

        base_score = 0.5

        # Longer messages indicate higher engagement
        length_score = min(len(message.content) / 500, 1.0) * 0.3

        # User messages have higher engagement than system messages
        role_score = 0.3 if message.role == MessageRole.USER else 0.1

        # Messages with metadata (tool results, etc.) are more important
        metadata_score = 0.2 if message.metadata else 0.0

        return base_score + length_score + role_score + metadata_score

    async def _calculate_task_relevance(
        self, message: Message, current_query: str
    ) -> float:
        """Calculate task relevance score"""

        # Check if message contains task-related keywords
        task_keywords = [
            "implement",
            "create",
            "generate",
            "analyze",
            "explain",
            "help",
        ]
        message_lower = message.content.lower()
        query_lower = current_query.lower()

        message_task_words = [word for word in task_keywords if word in message_lower]
        query_task_words = [word for word in task_keywords if word in query_lower]

        if not message_task_words and not query_task_words:
            return 0.5  # Neutral relevance

        common_task_words = set(message_task_words).intersection(set(query_task_words))
        return len(common_task_words) / max(len(query_task_words), 1)

    async def _generate_embedding(self, content: str) -> Optional[List[float]]:
        """Generate embedding vector for content"""

        try:
            # Check if we have embedding capability
            if (
                not hasattr(self, "_embedding_manager")
                or self._embedding_manager is None
            ):
                # Try to initialize embedding manager lazily
                try:
                    from src.rag_pipeline.embedding_manager import (
                        EmbeddingManager,
                        EmbeddingConfig,
                    )

                    self._embedding_manager = EmbeddingManager(EmbeddingConfig())
                    await self._embedding_manager.initialize()
                    logger.info("Embedding manager initialized for message embeddings")
                except Exception as e:
                    logger.debug(
                        f"Could not initialize embedding manager: {e}. Embeddings disabled."
                    )
                    self._embedding_manager = None
                    return None

            if self._embedding_manager:
                # Generate embedding
                embedding_result = await self._embedding_manager.generate_embeddings(
                    content
                )
                return embedding_result.general_embedding
            else:
                return None

        except Exception as e:
            logger.debug(f"Error generating embedding: {e}")
            return None

    async def _calculate_semantic_similarity(
        self, message: Message, current_query: str
    ) -> float:
        """Calculate semantic similarity using embeddings"""

        try:
            # Check if we have embedding capability
            if (
                not hasattr(self, "_embedding_manager")
                or self._embedding_manager is None
            ):
                # Try to initialize embedding manager lazily
                try:
                    from src.rag_pipeline.embedding_manager import (
                        EmbeddingManager,
                        EmbeddingConfig,
                    )

                    self._embedding_manager = EmbeddingManager(EmbeddingConfig())
                    await self._embedding_manager.initialize()
                    logger.info("Embedding manager initialized for semantic similarity")
                except Exception as e:
                    logger.warning(
                        f"Could not initialize embedding manager: {e}. Using keyword-based similarity."
                    )
                    self._embedding_manager = None

            if self._embedding_manager:
                # Generate embeddings for current query
                query_embedding_result = (
                    await self._embedding_manager.generate_embeddings(current_query)
                )
                query_embedding = query_embedding_result.general_embedding

                # Get or generate embedding for message
                if message.embedding:
                    message_embedding = message.embedding
                else:
                    # Generate embedding for message content
                    message_embedding_result = (
                        await self._embedding_manager.generate_embeddings(
                            message.content
                        )
                    )
                    message_embedding = message_embedding_result.general_embedding

                # Calculate cosine similarity
                import numpy as np

                query_vec = np.array(query_embedding)
                message_vec = np.array(message_embedding)

                # Cosine similarity
                dot_product = np.dot(query_vec, message_vec)
                norm_product = np.linalg.norm(query_vec) * np.linalg.norm(message_vec)

                if norm_product == 0:
                    return 0.0

                similarity = dot_product / norm_product
                # Normalize to 0-1 range (cosine similarity is -1 to 1)
                normalized_similarity = (similarity + 1) / 2

                return float(normalized_similarity)
            else:
                # Fallback: simple keyword-based similarity
                query_words = set(current_query.lower().split())
                message_words = set(message.content.lower().split())

                if not query_words or not message_words:
                    return 0.0

                # Jaccard similarity
                intersection = len(query_words & message_words)
                union = len(query_words | message_words)

                return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            # Fallback to neutral score
            return 0.5

    async def _maintain_conversation_flow(
        self, scored_messages: List[Tuple[Message, float]], target_size: int
    ) -> List[Message]:
        """Maintain conversation flow while selecting messages"""

        selected = []
        selected_indices = set()

        # Always include the most recent messages
        recent_count = min(3, target_size // 3)
        sorted_by_time = sorted(
            scored_messages, key=lambda x: x[0].timestamp, reverse=True
        )

        for i in range(recent_count):
            if i < len(sorted_by_time):
                selected.append(sorted_by_time[i][0])
                # Find index in original scored_messages
                for j, (msg, score) in enumerate(scored_messages):
                    if msg.message_id == sorted_by_time[i][0].message_id:
                        selected_indices.add(j)
                        break

        # Fill remaining slots with highest scoring messages
        remaining_slots = target_size - len(selected)
        for i, (msg, score) in enumerate(scored_messages):
            if i not in selected_indices and remaining_slots > 0:
                selected.append(msg)
                remaining_slots -= 1

        return selected


class ConversationSummarizer:
    """Creates intelligent summaries of conversation segments"""

    async def create_summary(
        self, messages: List[Message], session_id: str
    ) -> ConversationSummary:
        """Create a summary of a conversation segment"""

        if not messages:
            return ConversationSummary(
                session_id=session_id,
                summary_text="Empty conversation segment",
                key_topics=[],
                important_decisions=[],
                user_goals=[],
                created_at=datetime.utcnow(),
                covers_messages=0,
            )

        # Extract key information
        key_topics = await self._extract_key_topics(messages)
        important_decisions = await self._extract_decisions(messages)
        user_goals = await self._extract_user_goals(messages)

        # Generate summary text
        summary_text = await self._generate_summary_text(
            messages, key_topics, important_decisions, user_goals
        )

        # Calculate relevance score
        relevance_score = await self._calculate_summary_relevance(messages)

        return ConversationSummary(
            session_id=session_id,
            summary_text=summary_text,
            key_topics=key_topics,
            important_decisions=important_decisions,
            user_goals=user_goals,
            created_at=datetime.utcnow(),
            covers_messages=len(messages),
            relevance_score=relevance_score,
        )

    async def _extract_key_topics(self, messages: List[Message]) -> List[str]:
        """Extract key topics from messages"""

        topic_counts = defaultdict(int)

        # Simple keyword-based topic extraction
        topic_keywords = {
            "programming": ["code", "programming", "function", "class", "variable"],
            "documentation": ["document", "report", "write", "documentation"],
            "testing": ["test", "testing", "unit test", "debug"],
            "architecture": ["architecture", "design", "pattern", "structure"],
            "database": ["database", "sql", "query", "table"],
            "api": ["api", "endpoint", "rest", "graphql"],
            "deployment": ["deploy", "deployment", "server", "production"],
        }

        for message in messages:
            content_lower = message.content.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topic_counts[topic] += 1

        # Return top topics
        return [
            topic
            for topic, count in sorted(
                topic_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

    async def _extract_decisions(self, messages: List[Message]) -> List[str]:
        """Extract important decisions from messages"""

        decisions = []
        decision_indicators = [
            "decided",
            "choose",
            "selected",
            "will use",
            "going with",
        ]

        for message in messages:
            content_lower = message.content.lower()
            if any(indicator in content_lower for indicator in decision_indicators):
                # Extract the sentence containing the decision
                sentences = message.content.split(".")
                for sentence in sentences:
                    if any(
                        indicator in sentence.lower()
                        for indicator in decision_indicators
                    ):
                        decisions.append(sentence.strip())
                        break

        return decisions[:3]  # Return top 3 decisions

    async def _extract_user_goals(self, messages: List[Message]) -> List[str]:
        """Extract user goals from messages"""

        goals = []
        goal_indicators = ["want to", "need to", "trying to", "goal is", "objective"]

        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]

        for message in user_messages:
            content_lower = message.content.lower()
            if any(indicator in content_lower for indicator in goal_indicators):
                # Extract the goal statement
                sentences = message.content.split(".")
                for sentence in sentences:
                    if any(
                        indicator in sentence.lower() for indicator in goal_indicators
                    ):
                        goals.append(sentence.strip())
                        break

        return goals[:3]  # Return top 3 goals

    async def _generate_summary_text(
        self,
        messages: List[Message],
        key_topics: List[str],
        decisions: List[str],
        goals: List[str],
    ) -> str:
        """Generate summary text"""

        summary_parts = []

        if goals:
            summary_parts.append(f"User goals: {', '.join(goals[:2])}")

        if key_topics:
            summary_parts.append(f"Main topics discussed: {', '.join(key_topics[:3])}")

        if decisions:
            summary_parts.append(f"Key decisions: {', '.join(decisions[:2])}")

        message_count = len(messages)
        user_messages = len([msg for msg in messages if msg.role == MessageRole.USER])

        summary_parts.append(
            f"Conversation included {message_count} messages with {user_messages} user interactions"
        )

        return ". ".join(summary_parts) + "."

    async def _calculate_summary_relevance(self, messages: List[Message]) -> float:
        """Calculate relevance score for the summary"""

        # Base relevance
        relevance = 0.7

        # Boost for longer conversations
        if len(messages) > 10:
            relevance += 0.1

        # Boost for recent messages
        recent_messages = [
            msg for msg in messages if (datetime.utcnow() - msg.timestamp).days < 1
        ]
        if recent_messages:
            relevance += 0.2

        return min(relevance, 1.0)


class MemoryManager:
    """Enhanced memory manager with intelligent context management"""

    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None

        # In-memory storage (will be backed by Redis)
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.user_profiles: Dict[str, UserProfile] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        self.conversation_summaries: Dict[str, List[ConversationSummary]] = defaultdict(
            list
        )

        # Components
        self.context_pruner = ContextPruner()
        self.summarizer = ConversationSummarizer()

        # Configuration
        self.max_conversation_length = 1000
        self.short_term_window_size = 10
        self.working_memory_size = 20
        self.long_term_retention_days = 90
        self.summary_trigger_threshold = 50  # Create summary after 50 messages

        self._initialized = False

    async def initialize(self):
        """Initialize enhanced memory manager"""
        if self._initialized:
            return

        logger.info("Initializing Enhanced Memory Manager")

        try:
            # Initialize Redis connection
            redis_url = getattr(self.settings, "REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)

            # Test Redis connection
            await self.redis_client.ping()
            logger.info("Redis connection established for memory management")

            # Load existing data from Redis
            await self._load_from_persistent_storage()

            # Start background tasks
            asyncio.create_task(self._background_maintenance())

            self._initialized = True
            logger.info("Enhanced Memory Manager initialized successfully")

        except Exception as e:
            logger.warning(
                "Redis not available, using in-memory storage only", error=str(e)
            )
            self._initialized = True

    async def get_context(self, session_id: str) -> ConversationContext:
        """Get enhanced conversation context with intelligent pruning"""

        # Get all messages for the session
        all_messages = list(self.conversations.get(session_id, []))

        # Get user profile
        user_id = self.session_metadata.get(session_id, {}).get("user_id")
        user_profile = self.user_profiles.get(user_id) if user_id else None

        # Create context window with intelligent selection
        context_window = await self._create_context_window(
            all_messages, session_id, user_profile
        )

        # Convert to ConversationContext format
        message_history = [
            {
                "content": msg.content,
                "role": msg.role.value,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata,
                "relevance_score": msg.relevance_score,
                "context_tags": msg.context_tags,
            }
            for msg in context_window.messages
        ]

        # Get user preferences
        preferences = {}
        if user_profile:
            # user_profile.preferences is already a dict, no need for asdict()
            preferences = (
                user_profile.preferences if hasattr(user_profile, "preferences") else {}
            )

        # Determine current topic
        current_topic = await self._extract_current_topic_enhanced(
            context_window.messages
        )

        # Create enhanced conversation context
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            message_history=message_history,
            user_preferences=preferences,
            current_topic=current_topic,
            domain_context=await self._extract_domain_context(context_window.messages),
            conversation_state=context_window.session_state,
        )

        return context

    async def _create_context_window(
        self,
        all_messages: List[Message],
        session_id: str,
        user_profile: Optional[UserProfile],
    ) -> ContextWindow:
        """Create intelligent context window"""

        if not all_messages:
            return ContextWindow(
                messages=[],
                summary=None,
                relevant_history=[],
                user_context={},
                session_state={},
                total_relevance_score=0.0,
                window_size=0,
            )

        # Get recent messages for short-term context
        recent_messages = all_messages[-self.short_term_window_size :]

        # Get working memory (more messages with relevance filtering)
        working_memory_messages = []
        if len(all_messages) > self.short_term_window_size:
            # Get current query context (last user message)
            current_query = ""
            for msg in reversed(all_messages):
                if msg.role == MessageRole.USER:
                    current_query = msg.content
                    break

            # Prune context intelligently
            candidate_messages = all_messages[: -self.short_term_window_size]
            working_memory_messages = await self.context_pruner.prune_context(
                candidate_messages,
                current_query,
                self.working_memory_size,
                user_profile,
            )

        # Combine messages
        context_messages = working_memory_messages + recent_messages

        # Get relevant conversation summary
        summary = await self._get_relevant_summary(session_id, context_messages)

        # Extract user context
        user_context = await self._extract_user_context(context_messages, user_profile)

        # Extract session state
        session_state = await self._extract_session_state(context_messages)

        # Calculate total relevance score
        total_relevance = (
            sum(msg.relevance_score for msg in context_messages) / len(context_messages)
            if context_messages
            else 0.0
        )

        return ContextWindow(
            messages=context_messages,
            summary=summary,
            relevant_history=working_memory_messages,
            user_context=user_context,
            session_state=session_state,
            total_relevance_score=total_relevance,
            window_size=len(context_messages),
        )

    async def store_interaction(
        self,
        session_id: str,
        user_message: str,
        agent_result: ExecutionResult,
        user_id: Optional[str] = None,
    ):
        """Store interaction with enhanced metadata and analysis"""

        timestamp = datetime.utcnow()

        # Initialize session if needed
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = {
                "created_at": timestamp,
                "message_count": 0,
                "user_id": user_id,
            }

        if user_id:
            self.session_metadata[session_id]["user_id"] = user_id

            # Initialize user profile if needed
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id=user_id)

        # Analyze and enhance user message
        user_msg = await self._create_enhanced_message(
            content=user_message,
            role=MessageRole.USER,
            timestamp=timestamp,
            session_id=session_id,
            user_id=user_id,
        )

        # Analyze and enhance agent response
        agent_msg = await self._create_enhanced_message(
            content=agent_result.response,
            role=MessageRole.ASSISTANT,
            timestamp=timestamp,
            session_id=session_id,
            user_id=user_id,
            execution_metadata={
                "execution_time": agent_result.execution_time,
                "state": agent_result.state.value,
                "tool_results": agent_result.tool_results,
                **agent_result.metadata,
            },
        )

        # Store messages
        self.conversations[session_id].append(user_msg)
        self.conversations[session_id].append(agent_msg)

        # Update session metadata
        self.session_metadata[session_id]["message_count"] += 2
        self.session_metadata[session_id]["last_activity"] = timestamp

        # Update user profile
        if user_id:
            await self._update_user_profile(user_id, user_msg, agent_msg)

        # Check if we need to create a summary
        if len(self.conversations[session_id]) % self.summary_trigger_threshold == 0:
            await self._create_conversation_summary(session_id)

        # Persist to Redis if available
        if self.redis_client:
            await self._persist_to_redis(session_id, user_msg, agent_msg)

        logger.info(
            "Enhanced interaction stored",
            session_id=session_id,
            user_id=user_id,
            conversation_length=len(self.conversations[session_id]),
            user_relevance=user_msg.relevance_score,
            agent_relevance=agent_msg.relevance_score,
        )

    async def _create_enhanced_message(
        self,
        content: str,
        role: MessageRole,
        timestamp: datetime,
        session_id: str,
        user_id: Optional[str] = None,
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Create enhanced message with analysis and metadata"""

        # Extract context tags
        context_tags = await self._extract_context_tags(content)

        # Calculate relevance score
        relevance_score = await self._calculate_message_relevance_score(
            content, role, context_tags
        )

        # Create metadata
        metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "word_count": len(content.split()),
            "character_count": len(content),
            "context_tags": context_tags,
        }

        if execution_metadata:
            metadata.update(execution_metadata)

        # Generate embedding for semantic similarity
        try:
            embedding = await self._generate_embedding(content)
        except Exception as e:
            logger.warning(
                f"Failed to generate embedding for message: {e}. Continuing without embedding."
            )
            embedding = None

        return Message(
            content=content,
            role=role,
            timestamp=timestamp,
            metadata=metadata,
            relevance_score=relevance_score,
            context_tags=context_tags,
            embedding=embedding,
        )

    async def _extract_context_tags(self, content: str) -> List[str]:
        """Extract context tags from message content"""

        tags = []
        content_lower = content.lower()

        # Programming language tags
        languages = [
            "python",
            "javascript",
            "java",
            "c++",
            "c#",
            "go",
            "rust",
            "typescript",
        ]
        tags.extend([lang for lang in languages if lang in content_lower])

        # Framework tags
        frameworks = ["react", "angular", "vue", "django", "flask", "spring", "express"]
        tags.extend([fw for fw in frameworks if fw in content_lower])

        # Activity tags
        activities = {
            "coding": ["code", "implement", "function", "class"],
            "debugging": ["debug", "error", "fix", "bug"],
            "learning": ["learn", "understand", "explain", "how"],
            "planning": ["plan", "design", "architecture", "structure"],
            "testing": ["test", "unit", "integration", "validate"],
        }

        for activity, keywords in activities.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(activity)

        return list(set(tags))  # Remove duplicates

    async def _calculate_message_relevance_score(
        self, content: str, role: MessageRole, context_tags: List[str]
    ) -> float:
        """Calculate relevance score for a message"""

        base_score = 0.7

        # Role-based scoring
        if role == MessageRole.USER:
            base_score += 0.1  # User messages are generally more important
        elif role == MessageRole.TOOL:
            base_score += 0.05  # Tool results are moderately important

        # Content length scoring
        word_count = len(content.split())
        if word_count > 50:
            base_score += 0.1
        elif word_count < 5:
            base_score -= 0.1

        # Context tags scoring
        if context_tags:
            base_score += min(len(context_tags) * 0.05, 0.2)

        return min(base_score, 1.0)

    async def _update_user_profile(
        self, user_id: str, user_msg: Message, agent_msg: Message
    ):
        """Update user profile based on interaction"""

        profile = self.user_profiles[user_id]

        # Update frequent topics
        for tag in user_msg.context_tags:
            profile.frequent_topics[tag] = profile.frequent_topics.get(tag, 0) + 1

        # Update interaction patterns
        profile.interaction_patterns["total_messages"] = (
            profile.interaction_patterns.get("total_messages", 0) + 1
        )
        profile.interaction_patterns["avg_message_length"] = (
            profile.interaction_patterns.get("avg_message_length", 0) * 0.9
            + len(user_msg.content.split()) * 0.1
        )

        # Update expertise areas based on context tags
        for tag in user_msg.context_tags:
            if (
                tag not in profile.expertise_areas
                and profile.frequent_topics.get(tag, 0) > 5
            ):
                profile.expertise_areas.append(tag)

        profile.last_updated = datetime.utcnow()

    async def _create_conversation_summary(self, session_id: str):
        """Create summary for conversation segment"""

        messages = list(self.conversations[session_id])
        if len(messages) < self.summary_trigger_threshold:
            return

        # Get messages for summary (last segment)
        start_idx = max(0, len(messages) - self.summary_trigger_threshold)
        segment_messages = messages[start_idx:]

        # Create summary
        summary = await self.summarizer.create_summary(segment_messages, session_id)

        # Store summary
        self.conversation_summaries[session_id].append(summary)

        logger.info(
            "Conversation summary created",
            session_id=session_id,
            messages_summarized=len(segment_messages),
            key_topics=len(summary.key_topics),
        )

    async def _get_relevant_summary(
        self, session_id: str, context_messages: List[Message]
    ) -> Optional[ConversationSummary]:
        """Get most relevant conversation summary"""

        summaries = self.conversation_summaries.get(session_id, [])
        if not summaries:
            return None

        # If only one summary, return it
        if len(summaries) == 1:
            return summaries[0]

        # Implement relevance-based selection
        # Score summaries based on:
        # 1. Recency (more recent = higher score)
        # 2. Topic overlap with current context
        # 3. Key topics coverage

        # Extract topics from current context
        context_topics = set()
        for msg in context_messages[-5:]:  # Last 5 messages
            context_topics.update(msg.context_tags)

        best_summary = None
        best_score = -1

        for i, summary in enumerate(summaries):
            # Recency score (0-1, more recent = higher)
            recency_score = (i + 1) / len(summaries)

            # Topic overlap score
            summary_topics = set(summary.key_topics)
            if context_topics and summary_topics:
                topic_overlap = len(context_topics & summary_topics) / len(
                    context_topics | summary_topics
                )
            else:
                topic_overlap = 0.0

            # Combined score (weighted)
            combined_score = (0.6 * recency_score) + (0.4 * topic_overlap)

            if combined_score > best_score:
                best_score = combined_score
                best_summary = summary

        return best_summary if best_summary else summaries[-1]

    async def _extract_user_context(
        self, messages: List[Message], user_profile: Optional[UserProfile]
    ) -> Dict[str, Any]:
        """Extract user context from messages"""

        context = {}

        if user_profile:
            context["expertise_areas"] = user_profile.expertise_areas
            context["frequent_topics"] = user_profile.frequent_topics
            context["interaction_patterns"] = user_profile.interaction_patterns

        # Extract current session context
        recent_tags = []
        for msg in messages[-5:]:  # Last 5 messages
            recent_tags.extend(msg.context_tags)

        context["recent_topics"] = list(set(recent_tags))
        context["session_length"] = len(messages)

        return context

    async def _extract_session_state(self, messages: List[Message]) -> Dict[str, Any]:
        """Extract session state from messages"""

        state = {
            "last_user_intent": None,
            "pending_tasks": [],
            "completed_tasks": [],
            "current_focus": None,
        }

        # Extract last user intent
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                # Simple intent extraction
                content_lower = msg.content.lower()
                if any(word in content_lower for word in ["help", "explain", "what"]):
                    state["last_user_intent"] = "information_seeking"
                elif any(
                    word in content_lower for word in ["create", "generate", "build"]
                ):
                    state["last_user_intent"] = "creation"
                elif any(word in content_lower for word in ["fix", "debug", "error"]):
                    state["last_user_intent"] = "problem_solving"
                break

        # Extract current focus from recent context tags
        recent_tags = []
        for msg in messages[-3:]:
            recent_tags.extend(msg.context_tags)

        if recent_tags:
            # Most frequent recent tag is current focus
            from collections import Counter

            tag_counts = Counter(recent_tags)
            state["current_focus"] = tag_counts.most_common(1)[0][0]

        return state

    async def _extract_current_topic_enhanced(
        self, messages: List[Message]
    ) -> Optional[str]:
        """Enhanced topic extraction from messages"""

        if not messages:
            return None

        # Collect context tags from recent messages
        recent_tags = []
        for msg in messages[-5:]:
            recent_tags.extend(msg.context_tags)

        if recent_tags:
            # Return most frequent tag
            from collections import Counter

            tag_counts = Counter(recent_tags)
            return tag_counts.most_common(1)[0][0]

        # Fallback to simple keyword-based detection
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        if not user_messages:
            return None

        recent_content = user_messages[-1].content.lower()

        # Topic classification
        if any(
            word in recent_content
            for word in ["code", "programming", "function", "class"]
        ):
            return "programming"
        elif any(
            word in recent_content
            for word in ["document", "report", "write", "generate"]
        ):
            return "documentation"
        elif any(word in recent_content for word in ["explain", "what", "how", "why"]):
            return "explanation"
        elif any(word in recent_content for word in ["test", "testing", "unit test"]):
            return "testing"
        else:
            return "general"

    async def _extract_domain_context(self, messages: List[Message]) -> Dict[str, Any]:
        """Extract domain-specific context"""

        domain_context = {
            "primary_domain": "software_engineering",
            "sub_domains": [],
            "technical_level": "intermediate",
            "current_project_context": None,
        }

        # Extract sub-domains from context tags
        all_tags = []
        for msg in messages:
            all_tags.extend(msg.context_tags)

        # Map tags to sub-domains
        domain_mapping = {
            "web_development": ["javascript", "react", "angular", "vue", "html", "css"],
            "backend_development": ["python", "java", "spring", "django", "flask"],
            "mobile_development": ["android", "ios", "react native", "flutter"],
            "data_science": ["python", "pandas", "numpy", "machine learning"],
            "devops": ["docker", "kubernetes", "jenkins", "deployment"],
        }

        for domain, keywords in domain_mapping.items():
            if any(keyword in all_tags for keyword in keywords):
                domain_context["sub_domains"].append(domain)

        return domain_context

    async def _generate_embedding(self, content: str) -> Optional[List[float]]:
        """Generate embedding vector for content"""

        try:
            # Check if we have embedding capability
            if (
                not hasattr(self, "_embedding_manager")
                or self._embedding_manager is None
            ):
                # Try to initialize embedding manager lazily
                try:
                    from src.rag_pipeline.embedding_manager import (
                        EmbeddingManager,
                        EmbeddingConfig,
                    )

                    self._embedding_manager = EmbeddingManager(EmbeddingConfig())
                    await self._embedding_manager.initialize()
                    logger.info("Embedding manager initialized for message embeddings")
                except Exception as e:
                    logger.debug(
                        f"Could not initialize embedding manager: {e}. Embeddings disabled."
                    )
                    self._embedding_manager = None
                    return None

            if self._embedding_manager:
                # Generate embedding
                embedding_result = await self._embedding_manager.generate_embeddings(
                    content
                )
                return embedding_result.general_embedding
            else:
                return None

        except Exception as e:
            logger.debug(f"Error generating embedding: {e}")
            return None

    async def _persist_to_redis(
        self, session_id: str, user_msg: Message, agent_msg: Message
    ):
        """Persist messages to Redis"""

        if not self.redis_client:
            return

        try:
            # Store messages in Redis list
            user_data = {
                "content": user_msg.content,
                "role": user_msg.role.value,
                "timestamp": user_msg.timestamp.isoformat(),
                "metadata": user_msg.metadata,
                "relevance_score": user_msg.relevance_score,
                "context_tags": user_msg.context_tags,
            }

            agent_data = {
                "content": agent_msg.content,
                "role": agent_msg.role.value,
                "timestamp": agent_msg.timestamp.isoformat(),
                "metadata": agent_msg.metadata,
                "relevance_score": agent_msg.relevance_score,
                "context_tags": agent_msg.context_tags,
            }

            # Store in Redis
            await self.redis_client.lpush(
                f"conversation:{session_id}", json.dumps(user_data)
            )
            await self.redis_client.lpush(
                f"conversation:{session_id}", json.dumps(agent_data)
            )

            # Set expiration
            await self.redis_client.expire(
                f"conversation:{session_id}",
                60 * 60 * 24 * self.long_term_retention_days,
            )

        except Exception as e:
            logger.error("Failed to persist to Redis", error=str(e))

    async def _load_from_persistent_storage(self):
        """Load existing data from Redis"""

        if not self.redis_client:
            return

        try:
            logger.info("Loading existing conversations from Redis")

            # Get all session keys
            session_keys = await self.redis_client.keys("memory:session:*:messages")

            loaded_sessions = 0
            loaded_messages = 0

            for key in session_keys:
                try:
                    # Extract session_id from key
                    session_id = key.split(":")[2]

                    # Load messages for this session
                    message_data_list = await self.redis_client.lrange(key, 0, -1)

                    if not message_data_list:
                        continue

                    # Parse and reconstruct messages
                    messages = []
                    for msg_json in message_data_list:
                        import json

                        msg_data = json.loads(msg_json)

                        # Reconstruct Message object
                        message = Message(
                            content=msg_data["content"],
                            role=MessageRole(msg_data["role"]),
                            timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                            metadata=msg_data.get("metadata", {}),
                            relevance_score=msg_data.get("relevance_score", 0.5),
                            context_tags=msg_data.get("context_tags", []),
                            embedding=msg_data.get("embedding"),
                        )
                        messages.append(message)

                    # Store in memory
                    self.conversations[session_id] = messages
                    loaded_messages += len(messages)
                    loaded_sessions += 1

                    # Load session metadata
                    metadata_key = f"memory:session:{session_id}:metadata"
                    metadata_json = await self.redis_client.get(metadata_key)
                    if metadata_json:
                        import json

                        self.session_metadata[session_id] = json.loads(metadata_json)

                except Exception as e:
                    logger.warning(
                        f"Failed to load session {session_id} from Redis: {e}"
                    )
                    continue

            logger.info(
                f"Loaded {loaded_messages} messages from {loaded_sessions} sessions from Redis"
            )

        except Exception as e:
            logger.error("Failed to load from Redis", error=str(e))

    async def _background_maintenance(self):
        """Background maintenance tasks"""

        while True:
            try:
                # Clean up old conversations
                await self._cleanup_old_conversations()

                # Update user profiles
                await self._update_user_profiles_batch()

                # Create summaries for long conversations
                await self._create_pending_summaries()

                # Sleep for 1 hour
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error("Background maintenance error", error=str(e))
                await asyncio.sleep(300)  # Sleep 5 minutes on error

    async def _cleanup_old_conversations(self):
        """Clean up old conversations"""

        cutoff_date = datetime.utcnow() - timedelta(days=self.long_term_retention_days)
        sessions_to_remove = []

        for session_id, messages in self.conversations.items():
            if messages and messages[-1].timestamp < cutoff_date:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.conversations[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            if session_id in self.conversation_summaries:
                del self.conversation_summaries[session_id]

        if sessions_to_remove:
            logger.info("Cleaned up old conversations", count=len(sessions_to_remove))

    async def _update_user_profiles_batch(self):
        """Batch update user profiles"""

        for user_id, profile in self.user_profiles.items():
            # Update expertise areas based on frequent topics
            for topic, count in profile.frequent_topics.items():
                if count > 10 and topic not in profile.expertise_areas:
                    profile.expertise_areas.append(topic)

            profile.last_updated = datetime.utcnow()

    async def _create_pending_summaries(self):
        """Create summaries for conversations that need them"""

        for session_id, messages in self.conversations.items():
            if len(messages) >= self.summary_trigger_threshold:
                # Check if we need a new summary
                last_summary_count = 0
                if (
                    session_id in self.conversation_summaries
                    and self.conversation_summaries[session_id]
                ):
                    last_summary_count = self.conversation_summaries[session_id][
                        -1
                    ].covers_messages

                if len(messages) - last_summary_count >= self.summary_trigger_threshold:
                    await self._create_conversation_summary(session_id)

    # ... (keeping existing methods for compatibility)

    async def get_conversation_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""

        messages = list(self.conversations.get(session_id, []))

        if limit:
            messages = messages[-limit:]

        return [
            {
                "content": msg.content,
                "role": msg.role.value,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata,
                "relevance_score": msg.relevance_score,
                "context_tags": msg.context_tags,
            }
            for msg in messages
        ]

    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        profile = self.user_profiles[user_id]

        # Update preferences
        for key, value in preferences.items():
            if hasattr(profile.preferences, key):
                setattr(profile.preferences, key, value)
            else:
                profile.preferences[key] = value

        profile.last_updated = datetime.utcnow()

        logger.info(
            "User preferences updated", user_id=user_id, preferences=preferences
        )

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""

        if user_id in self.user_profiles:
            return self.user_profiles[user_id].preferences

        # Return default preferences
        return asdict(UserPreferences())

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get enhanced statistics for a session"""

        messages = list(self.conversations.get(session_id, []))

        if not messages:
            return {"message_count": 0, "duration": 0, "topics": []}

        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        assistant_messages = [
            msg for msg in messages if msg.role == MessageRole.ASSISTANT
        ]

        duration = (messages[-1].timestamp - messages[0].timestamp).total_seconds()

        # Extract topics from context tags
        all_tags = []
        for msg in messages:
            all_tags.extend(msg.context_tags)

        from collections import Counter

        topic_counts = Counter(all_tags)

        # Calculate average relevance
        avg_relevance = sum(msg.relevance_score for msg in messages) / len(messages)

        return {
            "message_count": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "duration_seconds": duration,
            "start_time": messages[0].timestamp.isoformat(),
            "last_activity": messages[-1].timestamp.isoformat(),
            "current_topic": await self._extract_current_topic_enhanced(messages[-5:]),
            "all_topics": list(topic_counts.keys()),
            "topic_distribution": dict(topic_counts.most_common(10)),
            "average_relevance_score": avg_relevance,
            "summaries_created": len(self.conversation_summaries.get(session_id, [])),
        }

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""

        total_conversations = len(self.conversations)
        total_messages = sum(len(messages) for messages in self.conversations.values())
        total_users = len(self.user_profiles)
        total_summaries = sum(
            len(summaries) for summaries in self.conversation_summaries.values()
        )

        # Calculate average conversation length
        avg_conversation_length = (
            total_messages / total_conversations if total_conversations > 0 else 0
        )

        # Active sessions (activity in last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        active_sessions = 0

        for session_id, messages in self.conversations.items():
            if messages and messages[-1].timestamp > cutoff_time:
                active_sessions += 1

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_users": total_users,
            "total_summaries": total_summaries,
            "average_conversation_length": avg_conversation_length,
            "active_sessions_24h": active_sessions,
            "memory_components": {
                "context_pruner": "active",
                "conversation_summarizer": "active",
                "user_profiling": "active",
                "redis_persistence": "active" if self.redis_client else "inactive",
            },
        }

    async def shutdown(self):
        """Shutdown enhanced memory manager"""
        logger.info("Shutting down Enhanced Memory Manager")

        # Persist final state to Redis
        if self.redis_client:
            try:
                # Persist all active conversations
                logger.info("Persisting final state to Redis")

                persist_count = 0
                for session_id, messages in self.conversations.items():
                    try:
                        # Persist messages
                        key = f"memory:session:{session_id}:messages"
                        import json

                        for msg in messages:
                            msg_data = {
                                "content": msg.content,
                                "role": msg.role.value,
                                "timestamp": msg.timestamp.isoformat(),
                                "metadata": msg.metadata,
                                "relevance_score": msg.relevance_score,
                                "context_tags": msg.context_tags,
                            }

                            # Convert embedding to list if it exists (numpy arrays aren't JSON serializable)
                            if msg.embedding is not None:
                                try:
                                    # Convert numpy array to list
                                    if hasattr(msg.embedding, "tolist"):
                                        msg_data["embedding"] = msg.embedding.tolist()
                                    elif isinstance(msg.embedding, list):
                                        msg_data["embedding"] = msg.embedding
                                    else:
                                        # Skip embedding if it's not a recognized type
                                        logger.debug(
                                            f"Skipping embedding of type {type(msg.embedding)}"
                                        )
                                except Exception as e:
                                    logger.debug(f"Failed to serialize embedding: {e}")

                            await self.redis_client.rpush(key, json.dumps(msg_data))

                        # Set expiration (30 days)
                        await self.redis_client.expire(key, 2592000)

                        # Persist session metadata
                        if session_id in self.session_metadata:
                            metadata_key = f"memory:session:{session_id}:metadata"

                            # Convert datetime objects to ISO format strings
                            metadata = self.session_metadata[session_id].copy()
                            for key, value in metadata.items():
                                if isinstance(value, datetime):
                                    metadata[key] = value.isoformat()

                            await self.redis_client.setex(
                                metadata_key,
                                2592000,
                                json.dumps(metadata),
                            )

                        persist_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to persist session {session_id}: {e}")

                logger.info(f"Persisted {persist_count} sessions to Redis")

                # Close Redis connection
                await self.redis_client.close()

            except Exception as e:
                logger.error("Error during final persistence", error=str(e))

        # Shutdown embedding manager if initialized
        if hasattr(self, "_embedding_manager") and self._embedding_manager:
            try:
                await self._embedding_manager.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down embedding manager: {e}")

        logger.info("Enhanced Memory Manager shutdown complete")
