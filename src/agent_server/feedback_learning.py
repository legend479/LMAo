"""
Feedback Learning System
Learn from human feedback to optimize prompts, strategies, and system parameters
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .feedback_system import Feedback, FeedbackAnalysis, FeedbackCategory, FeedbackType
from src.shared.logging import get_logger
from src.shared.llm.integration import get_llm_integration

logger = get_logger(__name__)


@dataclass
class PromptOptimization:
    """Optimized prompt based on feedback"""

    original_prompt: str
    optimized_prompt: str
    reasoning: str
    expected_improvement: float
    feedback_count: int


@dataclass
class StrategyAdjustment:
    """Strategy adjustment based on feedback"""

    strategy_name: str
    parameter: str
    old_value: Any
    new_value: Any
    reasoning: str
    confidence: float


@dataclass
class LearningInsights:
    """Insights learned from feedback"""

    timestamp: str
    feedback_analyzed: int
    prompt_optimizations: List[PromptOptimization]
    strategy_adjustments: List[StrategyAdjustment]
    quality_improvements: Dict[str, float]
    recommendations: List[str]


class FeedbackLearningSystem:
    """Learn from feedback to improve system performance"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Configuration
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.min_feedback_for_learning = self.config.get(
            "min_feedback_for_learning", 10
        )
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)

        # Components
        self.llm_integration = None

        # Learning history
        self.learning_history: List[LearningInsights] = []
        self.applied_optimizations: List[PromptOptimization] = []
        self.applied_adjustments: List[StrategyAdjustment] = []

        # Current optimized prompts
        self.optimized_prompts: Dict[str, str] = {}

        # Current strategy parameters
        self.optimized_parameters: Dict[str, Dict[str, Any]] = {}

        self._initialized = False

    async def initialize(self):
        """Initialize feedback learning system"""
        if self._initialized:
            return

        logger.info("Initializing Feedback Learning System")

        self.llm_integration = await get_llm_integration()

        # Load previous learning if available
        await self._load_learning_history()

        self._initialized = True
        logger.info("Feedback Learning System initialized")

    async def learn_from_feedback(
        self, feedback_list: List[Feedback], feedback_analysis: FeedbackAnalysis
    ) -> LearningInsights:
        """
        Learn from feedback and generate optimizations

        Args:
            feedback_list: List of feedback to learn from
            feedback_analysis: Analysis of feedback

        Returns:
            LearningInsights with optimizations and adjustments
        """
        if not self._initialized:
            await self.initialize()

        if len(feedback_list) < self.min_feedback_for_learning:
            logger.info(
                f"Insufficient feedback for learning: {len(feedback_list)} < {self.min_feedback_for_learning}"
            )
            return LearningInsights(
                timestamp=datetime.utcnow().isoformat(),
                feedback_analyzed=len(feedback_list),
                prompt_optimizations=[],
                strategy_adjustments=[],
                quality_improvements={},
                recommendations=[],
            )

        logger.info(f"Learning from {len(feedback_list)} feedback entries")

        # Generate prompt optimizations
        prompt_optimizations = await self._optimize_prompts(
            feedback_list, feedback_analysis
        )

        # Generate strategy adjustments
        strategy_adjustments = await self._adjust_strategies(
            feedback_list, feedback_analysis
        )

        # Calculate quality improvements
        quality_improvements = self._calculate_quality_improvements(feedback_analysis)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            feedback_analysis, prompt_optimizations, strategy_adjustments
        )

        # Create insights
        insights = LearningInsights(
            timestamp=datetime.utcnow().isoformat(),
            feedback_analyzed=len(feedback_list),
            prompt_optimizations=prompt_optimizations,
            strategy_adjustments=strategy_adjustments,
            quality_improvements=quality_improvements,
            recommendations=recommendations,
        )

        # Store in history
        self.learning_history.append(insights)

        # Save learning history
        await self._save_learning_history()

        logger.info(
            "Learning complete",
            prompt_optimizations=len(prompt_optimizations),
            strategy_adjustments=len(strategy_adjustments),
        )

        return insights

    async def _optimize_prompts(
        self, feedback_list: List[Feedback], feedback_analysis: FeedbackAnalysis
    ) -> List[PromptOptimization]:
        """Optimize prompts based on feedback"""

        optimizations = []

        # Identify queries with poor responses
        poor_responses = [
            fb
            for fb in feedback_list
            if fb.rating and fb.rating < 0.4 and fb.query and fb.response
        ]

        if not poor_responses:
            return optimizations

        # Group by query similarity
        query_groups = self._group_similar_queries(poor_responses)

        # Optimize prompts for each group
        for group in query_groups[:3]:  # Top 3 groups
            if len(group) < 3:  # Need at least 3 examples
                continue

            optimization = await self._optimize_prompt_for_group(group)
            if optimization:
                optimizations.append(optimization)

        return optimizations

    def _group_similar_queries(
        self, feedback_list: List[Feedback]
    ) -> List[List[Feedback]]:
        """Group similar queries together"""
        # Simple grouping by query length and keywords
        # Can be enhanced with embedding-based clustering

        groups = []
        used_indices = set()

        for i, fb1 in enumerate(feedback_list):
            if i in used_indices:
                continue

            group = [fb1]
            used_indices.add(i)

            for j, fb2 in enumerate(feedback_list[i + 1 :], start=i + 1):
                if j in used_indices:
                    continue

                # Simple similarity check
                if self._queries_similar(fb1.query, fb2.query):
                    group.append(fb2)
                    used_indices.add(j)

            if len(group) >= 2:
                groups.append(group)

        # Sort by group size
        groups.sort(key=len, reverse=True)

        return groups

    def _queries_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are similar"""
        # Simple similarity check
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard = intersection / union if union > 0 else 0.0

        return jaccard > 0.5

    async def _optimize_prompt_for_group(
        self, feedback_group: List[Feedback]
    ) -> Optional[PromptOptimization]:
        """Optimize prompt for a group of similar queries"""

        # Extract common patterns
        queries = [fb.query for fb in feedback_group]
        responses = [fb.response for fb in feedback_group]
        text_feedbacks = [fb.text_feedback for fb in feedback_group if fb.text_feedback]

        # Use LLM to optimize prompt
        try:
            system_prompt = """You are an expert at optimizing prompts for better AI responses.

Given examples of queries, poor responses, and user feedback, suggest an improved prompt template that would generate better responses.

Return a JSON object with:
{
    "optimized_prompt": "the improved prompt template",
    "reasoning": "why this prompt is better",
    "expected_improvement": 0.0-1.0
}"""

            user_prompt = f"""Analyze these examples and suggest a better prompt:

Queries:
{chr(10).join(f"- {q}" for q in queries[:5])}

Poor Responses:
{chr(10).join(f"- {r[:100]}..." for r in responses[:3])}

User Feedback:
{chr(10).join(f"- {f}" for f in text_feedbacks[:5])}

Suggest an optimized prompt template."""

            response = await self.llm_integration.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500,
            )

            result = json.loads(response)

            return PromptOptimization(
                original_prompt="[Current system prompt]",
                optimized_prompt=result["optimized_prompt"],
                reasoning=result["reasoning"],
                expected_improvement=result.get("expected_improvement", 0.5),
                feedback_count=len(feedback_group),
            )

        except Exception as e:
            logger.warning(f"Failed to optimize prompt with LLM: {e}")
            return None

    async def _adjust_strategies(
        self, feedback_list: List[Feedback], feedback_analysis: FeedbackAnalysis
    ) -> List[StrategyAdjustment]:
        """Adjust retrieval and generation strategies based on feedback"""

        adjustments = []

        # Adjust relevance threshold if retrieval quality is poor
        if feedback_analysis.retrieval_quality_avg < 0.5:
            adjustment = StrategyAdjustment(
                strategy_name="retrieval",
                parameter="relevance_threshold",
                old_value=0.3,
                new_value=0.4,
                reasoning="Retrieval quality is low, increasing relevance threshold to filter out low-quality results",
                confidence=0.8,
            )
            adjustments.append(adjustment)

        # Adjust context window if generation quality is poor
        if feedback_analysis.generation_quality_avg < 0.5:
            adjustment = StrategyAdjustment(
                strategy_name="context_optimization",
                parameter="max_tokens",
                old_value=4000,
                new_value=6000,
                reasoning="Generation quality is low, increasing context window to provide more information",
                confidence=0.7,
            )
            adjustments.append(adjustment)

        # Adjust query reformulation if many queries fail
        negative_ratio = feedback_analysis.negative_feedback / max(
            1, feedback_analysis.total_feedback
        )
        if negative_ratio > 0.5:
            adjustment = StrategyAdjustment(
                strategy_name="query_processing",
                parameter="enable_llm_reformulation",
                old_value=True,
                new_value=True,
                reasoning="High failure rate, ensuring query reformulation is enabled",
                confidence=0.9,
            )
            adjustments.append(adjustment)

        # Adjust reranking if relevance is inconsistent
        chunk_feedback = [
            fb
            for fb in feedback_list
            if fb.feedback_type == FeedbackType.CHUNK_RELEVANCE
        ]

        if len(chunk_feedback) > 5:
            relevant_chunks = sum(
                1 for fb in chunk_feedback if fb.rating and fb.rating > 0.5
            )
            relevance_rate = relevant_chunks / len(chunk_feedback)

            if relevance_rate < 0.6:
                adjustment = StrategyAdjustment(
                    strategy_name="search",
                    parameter="enable_reranking",
                    old_value=True,
                    new_value=True,
                    reasoning=f"Chunk relevance rate is {relevance_rate:.1%}, ensuring reranking is enabled",
                    confidence=0.85,
                )
                adjustments.append(adjustment)

        return adjustments

    def _calculate_quality_improvements(
        self, feedback_analysis: FeedbackAnalysis
    ) -> Dict[str, float]:
        """Calculate quality improvements over time"""

        if len(self.learning_history) < 2:
            return {}

        # Compare with previous analysis
        previous_insights = (
            self.learning_history[-2] if len(self.learning_history) >= 2 else None
        )

        if not previous_insights:
            return {}

        improvements = {}

        # Calculate improvement in each category
        # (This would compare with historical data)
        improvements["retrieval_quality"] = 0.0  # Placeholder
        improvements["generation_quality"] = 0.0  # Placeholder
        improvements["overall_satisfaction"] = 0.0  # Placeholder

        return improvements

    async def _generate_recommendations(
        self,
        feedback_analysis: FeedbackAnalysis,
        prompt_optimizations: List[PromptOptimization],
        strategy_adjustments: List[StrategyAdjustment],
    ) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        # From feedback analysis
        recommendations.extend(feedback_analysis.suggested_improvements)

        # From prompt optimizations
        if prompt_optimizations:
            recommendations.append(
                f"Apply {len(prompt_optimizations)} prompt optimizations to improve response quality"
            )

        # From strategy adjustments
        if strategy_adjustments:
            recommendations.append(
                f"Apply {len(strategy_adjustments)} strategy adjustments to improve performance"
            )

        # Based on common issues
        if "Irrelevant results" in feedback_analysis.common_issues:
            recommendations.append(
                "Focus on improving query understanding and relevance filtering"
            )

        if "Missing information" in feedback_analysis.common_issues:
            recommendations.append(
                "Expand knowledge base or improve multi-hop retrieval"
            )

        return recommendations[:5]

    async def apply_optimizations(
        self, insights: LearningInsights, auto_apply: bool = False
    ) -> Dict[str, Any]:
        """
        Apply learned optimizations

        Args:
            insights: Learning insights to apply
            auto_apply: Whether to apply automatically or return for review

        Returns:
            Application results
        """
        applied = {"prompts": [], "strategies": [], "success": True, "errors": []}

        # Apply prompt optimizations
        for optimization in insights.prompt_optimizations:
            if auto_apply or optimization.expected_improvement > 0.7:
                try:
                    # Store optimized prompt
                    prompt_key = f"optimized_prompt_{len(self.optimized_prompts)}"
                    self.optimized_prompts[prompt_key] = optimization.optimized_prompt
                    self.applied_optimizations.append(optimization)
                    applied["prompts"].append(prompt_key)

                    logger.info(
                        "Applied prompt optimization",
                        key=prompt_key,
                        expected_improvement=optimization.expected_improvement,
                    )
                except Exception as e:
                    applied["errors"].append(
                        f"Failed to apply prompt optimization: {e}"
                    )
                    applied["success"] = False

        # Apply strategy adjustments
        for adjustment in insights.strategy_adjustments:
            if auto_apply or adjustment.confidence > self.confidence_threshold:
                try:
                    # Store adjusted parameter
                    if adjustment.strategy_name not in self.optimized_parameters:
                        self.optimized_parameters[adjustment.strategy_name] = {}

                    self.optimized_parameters[adjustment.strategy_name][
                        adjustment.parameter
                    ] = adjustment.new_value
                    self.applied_adjustments.append(adjustment)
                    applied["strategies"].append(
                        f"{adjustment.strategy_name}.{adjustment.parameter}"
                    )

                    logger.info(
                        "Applied strategy adjustment",
                        strategy=adjustment.strategy_name,
                        parameter=adjustment.parameter,
                        new_value=adjustment.new_value,
                    )
                except Exception as e:
                    applied["errors"].append(
                        f"Failed to apply strategy adjustment: {e}"
                    )
                    applied["success"] = False

        return applied

    def get_optimized_prompt(self, prompt_key: str) -> Optional[str]:
        """Get optimized prompt by key"""
        return self.optimized_prompts.get(prompt_key)

    def get_optimized_parameter(
        self, strategy_name: str, parameter: str
    ) -> Optional[Any]:
        """Get optimized parameter value"""
        if strategy_name in self.optimized_parameters:
            return self.optimized_parameters[strategy_name].get(parameter)
        return None

    async def _save_learning_history(self):
        """Save learning history to file"""
        try:
            history_file = "learning_history.json"
            history_data = []

            for insights in self.learning_history:
                history_data.append(
                    {
                        "timestamp": insights.timestamp,
                        "feedback_analyzed": insights.feedback_analyzed,
                        "prompt_optimizations": len(insights.prompt_optimizations),
                        "strategy_adjustments": len(insights.strategy_adjustments),
                        "quality_improvements": insights.quality_improvements,
                        "recommendations": insights.recommendations,
                    }
                )

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            logger.debug(f"Saved learning history: {len(history_data)} entries")
        except Exception as e:
            logger.error(f"Failed to save learning history: {e}")

    async def _load_learning_history(self):
        """Load learning history from file"""
        try:
            history_file = "learning_history.json"
            with open(history_file, "r") as f:
                history_data = json.load(f)

            logger.info(f"Loaded learning history: {len(history_data)} entries")
        except FileNotFoundError:
            logger.info("No existing learning history found")
        except Exception as e:
            logger.error(f"Failed to load learning history: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for feedback learning system"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "learning_history_entries": len(self.learning_history),
            "applied_optimizations": len(self.applied_optimizations),
            "applied_adjustments": len(self.applied_adjustments),
            "optimized_prompts": len(self.optimized_prompts),
            "optimized_parameters": len(self.optimized_parameters),
        }
