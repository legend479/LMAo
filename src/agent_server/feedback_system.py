"""
Human Feedback System
Collect, analyze, and learn from human feedback to improve system performance
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json

from src.shared.logging import get_logger
from src.shared.llm.integration import get_llm_integration

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RELEVANCE_RATING = "relevance_rating"  # 1-5 stars
    DETAILED_FEEDBACK = "detailed_feedback"  # Text feedback
    MISSING_INFO = "missing_info"  # What was missing
    INCORRECT_INFO = "incorrect_info"  # What was wrong
    CHUNK_RELEVANCE = "chunk_relevance"  # Per-chunk feedback


class FeedbackCategory(Enum):
    """Categories of feedback"""

    RETRIEVAL_QUALITY = "retrieval_quality"
    GENERATION_QUALITY = "generation_quality"
    TOOL_PERFORMANCE = "tool_performance"
    OVERALL_SATISFACTION = "overall_satisfaction"


@dataclass
class Feedback:
    """Individual feedback entry"""

    feedback_id: str
    session_id: str
    user_id: Optional[str]
    timestamp: str

    # Feedback content
    feedback_type: FeedbackType
    category: FeedbackCategory
    rating: Optional[float] = None  # 0-1 normalized
    text_feedback: Optional[str] = None

    # Context
    query: str = ""
    response: str = ""
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["feedback_type"] = self.feedback_type.value
        data["category"] = self.category.value
        return data


@dataclass
class FeedbackAnalysis:
    """Analysis of collected feedback"""

    total_feedback: int
    positive_feedback: int
    negative_feedback: int
    average_rating: float

    # Category-specific metrics
    retrieval_quality_avg: float
    generation_quality_avg: float
    tool_performance_avg: float
    overall_satisfaction_avg: float

    # Common issues
    common_issues: List[str]
    missing_info_patterns: List[str]
    incorrect_info_patterns: List[str]

    # Improvement suggestions
    suggested_improvements: List[str]


class FeedbackCollector:
    """Collect and store human feedback"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Storage
        self.feedback_store: List[Feedback] = []
        self.feedback_by_session: Dict[str, List[Feedback]] = {}
        self.feedback_by_category: Dict[FeedbackCategory, List[Feedback]] = {
            category: [] for category in FeedbackCategory
        }

        # Configuration
        self.max_feedback_age_days = self.config.get("max_feedback_age_days", 30)
        self.enable_persistence = self.config.get("enable_persistence", True)
        self.feedback_file = self.config.get("feedback_file", "feedback_data.json")

        self._initialized = False

    async def initialize(self):
        """Initialize feedback collector"""
        if self._initialized:
            return

        logger.info("Initializing Feedback Collector")

        # Load existing feedback if persistence enabled
        if self.enable_persistence:
            await self._load_feedback()

        self._initialized = True
        logger.info(
            f"Feedback Collector initialized with {len(self.feedback_store)} existing feedback entries"
        )

    async def collect_feedback(
        self,
        session_id: str,
        feedback_type: FeedbackType,
        category: FeedbackCategory,
        query: str,
        response: str,
        rating: Optional[float] = None,
        text_feedback: Optional[str] = None,
        user_id: Optional[str] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        tools_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Feedback:
        """
        Collect feedback from user

        Args:
            session_id: Session identifier
            feedback_type: Type of feedback
            category: Feedback category
            query: Original query
            response: System response
            rating: Optional rating (0-1)
            text_feedback: Optional text feedback
            user_id: Optional user identifier
            retrieved_chunks: Optional retrieved chunks
            tools_used: Optional tools used
            metadata: Optional additional metadata

        Returns:
            Feedback object
        """
        import uuid

        feedback = Feedback(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            feedback_type=feedback_type,
            category=category,
            rating=rating,
            text_feedback=text_feedback,
            query=query,
            response=response,
            retrieved_chunks=retrieved_chunks or [],
            tools_used=tools_used or [],
            metadata=metadata or {},
        )

        # Store feedback
        self.feedback_store.append(feedback)

        # Index by session
        if session_id not in self.feedback_by_session:
            self.feedback_by_session[session_id] = []
        self.feedback_by_session[session_id].append(feedback)

        # Index by category
        self.feedback_by_category[category].append(feedback)

        # Persist if enabled
        if self.enable_persistence:
            await self._save_feedback()

        logger.info(
            "Feedback collected",
            feedback_id=feedback.feedback_id,
            type=feedback_type.value,
            category=category.value,
            rating=rating,
        )

        return feedback

    async def collect_thumbs_feedback(
        self, session_id: str, query: str, response: str, is_positive: bool, **kwargs
    ) -> Feedback:
        """Collect simple thumbs up/down feedback"""
        return await self.collect_feedback(
            session_id=session_id,
            feedback_type=(
                FeedbackType.THUMBS_UP if is_positive else FeedbackType.THUMBS_DOWN
            ),
            category=FeedbackCategory.OVERALL_SATISFACTION,
            query=query,
            response=response,
            rating=1.0 if is_positive else 0.0,
            **kwargs,
        )

    async def collect_rating_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        rating: float,  # 0-1
        category: FeedbackCategory = FeedbackCategory.OVERALL_SATISFACTION,
        **kwargs,
    ) -> Feedback:
        """Collect rating feedback (0-1 scale)"""
        return await self.collect_feedback(
            session_id=session_id,
            feedback_type=FeedbackType.RELEVANCE_RATING,
            category=category,
            query=query,
            response=response,
            rating=rating,
            **kwargs,
        )

    async def collect_chunk_feedback(
        self, session_id: str, query: str, chunk_id: str, is_relevant: bool, **kwargs
    ) -> Feedback:
        """Collect feedback on individual chunk relevance"""
        return await self.collect_feedback(
            session_id=session_id,
            feedback_type=FeedbackType.CHUNK_RELEVANCE,
            category=FeedbackCategory.RETRIEVAL_QUALITY,
            query=query,
            response="",
            rating=1.0 if is_relevant else 0.0,
            metadata={"chunk_id": chunk_id, "is_relevant": is_relevant},
            **kwargs,
        )

    def get_feedback_by_session(self, session_id: str) -> List[Feedback]:
        """Get all feedback for a session"""
        return self.feedback_by_session.get(session_id, [])

    def get_feedback_by_category(self, category: FeedbackCategory) -> List[Feedback]:
        """Get all feedback for a category"""
        return self.feedback_by_category.get(category, [])

    def get_recent_feedback(self, days: int = 7) -> List[Feedback]:
        """Get feedback from last N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)

        return [
            fb
            for fb in self.feedback_store
            if datetime.fromisoformat(fb.timestamp) > cutoff
        ]

    async def _save_feedback(self):
        """Save feedback to file"""
        try:
            feedback_data = [fb.to_dict() for fb in self.feedback_store]
            with open(self.feedback_file, "w") as f:
                json.dump(feedback_data, f, indent=2)
            logger.debug(f"Saved {len(feedback_data)} feedback entries")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    async def _load_feedback(self):
        """Load feedback from file"""
        try:
            with open(self.feedback_file, "r") as f:
                feedback_data = json.load(f)

            for data in feedback_data:
                # Reconstruct Feedback object
                feedback = Feedback(
                    feedback_id=data["feedback_id"],
                    session_id=data["session_id"],
                    user_id=data.get("user_id"),
                    timestamp=data["timestamp"],
                    feedback_type=FeedbackType(data["feedback_type"]),
                    category=FeedbackCategory(data["category"]),
                    rating=data.get("rating"),
                    text_feedback=data.get("text_feedback"),
                    query=data.get("query", ""),
                    response=data.get("response", ""),
                    retrieved_chunks=data.get("retrieved_chunks", []),
                    tools_used=data.get("tools_used", []),
                    metadata=data.get("metadata", {}),
                )

                self.feedback_store.append(feedback)

                # Index
                if feedback.session_id not in self.feedback_by_session:
                    self.feedback_by_session[feedback.session_id] = []
                self.feedback_by_session[feedback.session_id].append(feedback)

                self.feedback_by_category[feedback.category].append(feedback)

            logger.info(f"Loaded {len(feedback_data)} feedback entries")
        except FileNotFoundError:
            logger.info("No existing feedback file found")
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for feedback collector"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "total_feedback": len(self.feedback_store),
            "feedback_by_category": {
                category.value: len(feedbacks)
                for category, feedbacks in self.feedback_by_category.items()
            },
            "recent_feedback_7d": len(self.get_recent_feedback(7)),
        }


class FeedbackAnalyzer:
    """Analyze feedback to extract insights"""

    def __init__(self):
        self.llm_integration = None
        self._initialized = False

    async def initialize(self):
        """Initialize feedback analyzer"""
        if self._initialized:
            return

        logger.info("Initializing Feedback Analyzer")
        self.llm_integration = await get_llm_integration()
        self._initialized = True
        logger.info("Feedback Analyzer initialized")

    async def analyze_feedback(
        self, feedback_list: List[Feedback], time_period_days: int = 7
    ) -> FeedbackAnalysis:
        """
        Analyze feedback to extract insights

        Args:
            feedback_list: List of feedback to analyze
            time_period_days: Time period for analysis

        Returns:
            FeedbackAnalysis with insights
        """
        if not feedback_list:
            return FeedbackAnalysis(
                total_feedback=0,
                positive_feedback=0,
                negative_feedback=0,
                average_rating=0.0,
                retrieval_quality_avg=0.0,
                generation_quality_avg=0.0,
                tool_performance_avg=0.0,
                overall_satisfaction_avg=0.0,
                common_issues=[],
                missing_info_patterns=[],
                incorrect_info_patterns=[],
                suggested_improvements=[],
            )

        # Basic metrics
        total_feedback = len(feedback_list)
        positive_feedback = sum(
            1
            for fb in feedback_list
            if fb.feedback_type == FeedbackType.THUMBS_UP
            or (fb.rating and fb.rating >= 0.6)
        )
        negative_feedback = sum(
            1
            for fb in feedback_list
            if fb.feedback_type == FeedbackType.THUMBS_DOWN
            or (fb.rating and fb.rating < 0.4)
        )

        # Average rating
        ratings = [fb.rating for fb in feedback_list if fb.rating is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else 0.0

        # Category-specific averages
        category_averages = {}
        for category in FeedbackCategory:
            category_feedback = [fb for fb in feedback_list if fb.category == category]
            category_ratings = [
                fb.rating for fb in category_feedback if fb.rating is not None
            ]
            category_averages[category] = (
                sum(category_ratings) / len(category_ratings)
                if category_ratings
                else 0.0
            )

        # Extract common issues
        common_issues = await self._extract_common_issues(feedback_list)

        # Extract missing info patterns
        missing_info_patterns = await self._extract_missing_info_patterns(feedback_list)

        # Extract incorrect info patterns
        incorrect_info_patterns = await self._extract_incorrect_info_patterns(
            feedback_list
        )

        # Generate improvement suggestions
        suggested_improvements = await self._generate_improvement_suggestions(
            feedback_list, common_issues, missing_info_patterns, incorrect_info_patterns
        )

        return FeedbackAnalysis(
            total_feedback=total_feedback,
            positive_feedback=positive_feedback,
            negative_feedback=negative_feedback,
            average_rating=average_rating,
            retrieval_quality_avg=category_averages.get(
                FeedbackCategory.RETRIEVAL_QUALITY, 0.0
            ),
            generation_quality_avg=category_averages.get(
                FeedbackCategory.GENERATION_QUALITY, 0.0
            ),
            tool_performance_avg=category_averages.get(
                FeedbackCategory.TOOL_PERFORMANCE, 0.0
            ),
            overall_satisfaction_avg=category_averages.get(
                FeedbackCategory.OVERALL_SATISFACTION, 0.0
            ),
            common_issues=common_issues,
            missing_info_patterns=missing_info_patterns,
            incorrect_info_patterns=incorrect_info_patterns,
            suggested_improvements=suggested_improvements,
        )

    async def _extract_common_issues(self, feedback_list: List[Feedback]) -> List[str]:
        """Extract common issues from feedback"""
        issues = []

        # Collect text feedback
        text_feedbacks = [
            fb.text_feedback
            for fb in feedback_list
            if fb.text_feedback and fb.rating and fb.rating < 0.6
        ]

        if not text_feedbacks:
            return issues

        # Use LLM to extract common themes
        try:
            system_prompt = """You are an expert at analyzing user feedback.
Extract the top 5 common issues or complaints from the feedback provided.
Return as a JSON array of strings."""

            user_prompt = f"""Analyze these user feedback comments and extract common issues:

{chr(10).join(f"- {fb}" for fb in text_feedbacks[:20])}

Extract the top 5 common issues."""

            response = await self.llm_integration.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500,
            )

            issues = json.loads(response)
        except Exception as e:
            logger.warning(f"Failed to extract common issues with LLM: {e}")
            # Fallback: simple keyword extraction
            issues = self._extract_issues_fallback(text_feedbacks)

        return issues[:5]

    def _extract_issues_fallback(self, text_feedbacks: List[str]) -> List[str]:
        """Fallback method for extracting issues"""
        issue_keywords = {
            "not relevant": "Irrelevant results",
            "missing": "Missing information",
            "incorrect": "Incorrect information",
            "slow": "Slow response time",
            "confusing": "Confusing response",
        }

        issue_counts = {issue: 0 for issue in issue_keywords.values()}

        for feedback in text_feedbacks:
            feedback_lower = feedback.lower()
            for keyword, issue in issue_keywords.items():
                if keyword in feedback_lower:
                    issue_counts[issue] += 1

        # Return top issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues if count > 0][:5]

    async def _extract_missing_info_patterns(
        self, feedback_list: List[Feedback]
    ) -> List[str]:
        """Extract patterns of missing information"""
        missing_info_feedback = [
            fb for fb in feedback_list if fb.feedback_type == FeedbackType.MISSING_INFO
        ]

        if not missing_info_feedback:
            return []

        patterns = []
        for fb in missing_info_feedback[:10]:
            if fb.text_feedback:
                patterns.append(fb.text_feedback)

        return patterns

    async def _extract_incorrect_info_patterns(
        self, feedback_list: List[Feedback]
    ) -> List[str]:
        """Extract patterns of incorrect information"""
        incorrect_info_feedback = [
            fb
            for fb in feedback_list
            if fb.feedback_type == FeedbackType.INCORRECT_INFO
        ]

        if not incorrect_info_feedback:
            return []

        patterns = []
        for fb in incorrect_info_feedback[:10]:
            if fb.text_feedback:
                patterns.append(fb.text_feedback)

        return patterns

    async def _generate_improvement_suggestions(
        self,
        feedback_list: List[Feedback],
        common_issues: List[str],
        missing_info_patterns: List[str],
        incorrect_info_patterns: List[str],
    ) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []

        # Based on common issues
        if "Irrelevant results" in common_issues:
            suggestions.append("Improve query understanding and reformulation")
            suggestions.append("Adjust relevance thresholds")

        if "Missing information" in common_issues:
            suggestions.append("Expand knowledge base coverage")
            suggestions.append("Improve multi-hop retrieval")

        if "Incorrect information" in common_issues:
            suggestions.append("Add fact-checking mechanisms")
            suggestions.append("Improve source verification")

        if "Slow response time" in common_issues:
            suggestions.append("Optimize retrieval pipeline")
            suggestions.append("Implement better caching")

        # Based on ratings
        avg_retrieval = sum(
            fb.rating
            for fb in feedback_list
            if fb.category == FeedbackCategory.RETRIEVAL_QUALITY and fb.rating
        ) / max(
            1,
            len(
                [
                    fb
                    for fb in feedback_list
                    if fb.category == FeedbackCategory.RETRIEVAL_QUALITY and fb.rating
                ]
            ),
        )

        if avg_retrieval < 0.6:
            suggestions.append("Focus on improving retrieval quality")

        return suggestions[:5]

    async def health_check(self) -> Dict[str, Any]:
        """Health check for feedback analyzer"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "llm_available": self.llm_integration is not None,
        }
