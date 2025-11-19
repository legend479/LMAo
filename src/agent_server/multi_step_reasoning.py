"""
Multi-Step Reasoning Engine
Enables iterative multi-tool calls and complex reasoning procedures
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .orchestrator import ExecutionPlan, WorkflowState
from .adaptive_planning import AdaptivePlanningEngine, TaskVerificationResult
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ReasoningStepType(Enum):
    """Types of reasoning steps"""

    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    REFINEMENT = "refinement"
    TOOL_EXECUTION = "tool_execution"
    DECISION = "decision"


@dataclass
class ReasoningStep:
    """A single step in multi-step reasoning"""

    step_id: str
    step_type: ReasoningStepType
    description: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    completed: bool = False
    verified: bool = False


@dataclass
class ReasoningChain:
    """A chain of reasoning steps"""

    chain_id: str
    goal: str
    steps: List[ReasoningStep]
    current_step_index: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False


class MultiStepReasoningEngine:
    """Engine for multi-step reasoning and iterative tool execution"""

    def __init__(self, orchestrator, adaptive_engine: AdaptivePlanningEngine):
        self.orchestrator = orchestrator
        self.adaptive_engine = adaptive_engine
        self.active_chains: Dict[str, ReasoningChain] = {}

    async def create_reasoning_chain(
        self, goal: str, initial_context: Dict[str, Any], session_id: str
    ) -> ReasoningChain:
        """Create a multi-step reasoning chain for a complex goal"""

        logger.info(f"Creating reasoning chain for goal: {goal}")

        # Analyze the goal to determine required steps
        steps = await self._decompose_goal_into_reasoning_steps(goal, initial_context)

        chain = ReasoningChain(
            chain_id=f"chain_{session_id}_{len(self.active_chains)}",
            goal=goal,
            steps=steps,
            context=initial_context,
        )

        self.active_chains[chain.chain_id] = chain

        logger.info(
            f"Reasoning chain created with {len(steps)} steps", chain_id=chain.chain_id
        )

        return chain

    async def execute_reasoning_chain(
        self, chain: ReasoningChain, session_id: str
    ) -> Dict[str, Any]:
        """Execute a reasoning chain with iterative refinement"""

        logger.info(f"Executing reasoning chain: {chain.chain_id}")

        results = []

        while chain.current_step_index < len(chain.steps):
            step = chain.steps[chain.current_step_index]

            logger.info(
                f"Executing step {chain.current_step_index + 1}/{len(chain.steps)}",
                step_type=step.step_type.value,
                description=step.description,
            )

            # Check dependencies
            if not await self._check_dependencies(step, chain):
                logger.warning(f"Dependencies not met for step {step.step_id}")
                break

            # Execute the step
            step_result = await self._execute_reasoning_step(step, chain, session_id)

            # Verify the step
            verification = await self._verify_reasoning_step(step, step_result, chain)

            if verification.success:
                step.result = step_result
                step.completed = True
                step.verified = True
                results.append(step_result)

                # Update chain context with step results
                chain.context[f"step_{step.step_id}_result"] = step_result

                chain.current_step_index += 1
            else:
                # Step failed verification - attempt refinement
                logger.warning(
                    f"Step {step.step_id} failed verification",
                    issues=verification.issues_found,
                )

                if verification.retry_recommended:
                    # Refine and retry
                    refined_step = await self._refine_reasoning_step(
                        step, verification, chain
                    )

                    # Retry the refined step
                    retry_result = await self._execute_reasoning_step(
                        refined_step, chain, session_id
                    )

                    # Verify retry
                    retry_verification = await self._verify_reasoning_step(
                        refined_step, retry_result, chain
                    )

                    if retry_verification.success:
                        step.result = retry_result
                        step.completed = True
                        step.verified = True
                        results.append(retry_result)
                        chain.context[f"step_{step.step_id}_result"] = retry_result
                        chain.current_step_index += 1
                    else:
                        # Retry failed, skip or abort
                        logger.error(f"Step {step.step_id} failed after retry")
                        break
                else:
                    # Not recommended to retry, skip
                    logger.error(f"Skipping step {step.step_id}")
                    break

        chain.completed = chain.current_step_index >= len(chain.steps)

        return {
            "chain_id": chain.chain_id,
            "goal": chain.goal,
            "completed": chain.completed,
            "steps_completed": chain.current_step_index,
            "total_steps": len(chain.steps),
            "results": results,
            "final_context": chain.context,
        }

    async def _decompose_goal_into_reasoning_steps(
        self, goal: str, context: Dict[str, Any]
    ) -> List[ReasoningStep]:
        """Decompose a complex goal into reasoning steps"""

        steps = []

        # Use LLM to decompose goal
        if self.orchestrator.llm_integration:
            decomposition_prompt = f"""Break down this goal into logical reasoning steps:

Goal: {goal}

Context: {context}

Provide a step-by-step plan with:
1. Analysis steps (understand the problem)
2. Tool execution steps (gather information)
3. Synthesis steps (combine information)
4. Verification steps (validate results)

Format each step as:
- Step type: [analysis/tool_execution/synthesis/verification]
- Description: [what to do]
- Tool (if applicable): [tool name]
"""

            try:
                decomposition = (
                    await self.orchestrator.llm_integration.generate_response(
                        prompt=decomposition_prompt, temperature=0.3, max_tokens=1000
                    )
                )

                # Parse decomposition into steps
                steps = self._parse_decomposition(decomposition)

            except Exception as e:
                logger.error(f"Failed to decompose goal: {e}")

        # Fallback: Create basic steps
        if not steps:
            steps = [
                ReasoningStep(
                    step_id="step_1",
                    step_type=ReasoningStepType.ANALYSIS,
                    description=f"Analyze goal: {goal}",
                    success_criteria=["Goal understood"],
                ),
                ReasoningStep(
                    step_id="step_2",
                    step_type=ReasoningStepType.TOOL_EXECUTION,
                    description="Execute required tools",
                    success_criteria=["Tools executed successfully"],
                ),
                ReasoningStep(
                    step_id="step_3",
                    step_type=ReasoningStepType.SYNTHESIS,
                    description="Synthesize results",
                    success_criteria=["Results combined coherently"],
                ),
            ]

        return steps

    def _parse_decomposition(self, decomposition: str) -> List[ReasoningStep]:
        """Parse LLM decomposition into reasoning steps"""

        steps = []
        lines = decomposition.split("\n")

        current_step = None
        step_counter = 1

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect step type
            if "analysis" in line.lower():
                if current_step:
                    steps.append(current_step)
                current_step = ReasoningStep(
                    step_id=f"step_{step_counter}",
                    step_type=ReasoningStepType.ANALYSIS,
                    description=line,
                )
                step_counter += 1
            elif "tool" in line.lower() or "execute" in line.lower():
                if current_step:
                    steps.append(current_step)
                current_step = ReasoningStep(
                    step_id=f"step_{step_counter}",
                    step_type=ReasoningStepType.TOOL_EXECUTION,
                    description=line,
                )
                step_counter += 1
            elif "synthesis" in line.lower() or "combine" in line.lower():
                if current_step:
                    steps.append(current_step)
                current_step = ReasoningStep(
                    step_id=f"step_{step_counter}",
                    step_type=ReasoningStepType.SYNTHESIS,
                    description=line,
                )
                step_counter += 1
            elif "verif" in line.lower() or "validate" in line.lower():
                if current_step:
                    steps.append(current_step)
                current_step = ReasoningStep(
                    step_id=f"step_{step_counter}",
                    step_type=ReasoningStepType.VERIFICATION,
                    description=line,
                )
                step_counter += 1

        if current_step:
            steps.append(current_step)

        return steps

    async def _check_dependencies(
        self, step: ReasoningStep, chain: ReasoningChain
    ) -> bool:
        """Check if step dependencies are satisfied"""

        for dep_id in step.dependencies:
            dep_result = chain.context.get(f"step_{dep_id}_result")
            if not dep_result:
                return False

        return True

    async def _execute_reasoning_step(
        self, step: ReasoningStep, chain: ReasoningChain, session_id: str
    ) -> Dict[str, Any]:
        """Execute a single reasoning step"""

        if step.step_type == ReasoningStepType.TOOL_EXECUTION:
            # Execute tool
            if step.tool_name and self.orchestrator.tool_registry:
                from .tools.registry import ExecutionContext, ExecutionPriority

                context = ExecutionContext(
                    session_id=session_id,
                    priority=ExecutionPriority.NORMAL,
                    timeout=300,
                )

                tool = await self.orchestrator.tool_registry.get_tool_by_name(
                    step.tool_name
                )

                if tool:
                    result = await tool.execute(step.parameters, context)
                    return {
                        "type": "tool_execution",
                        "tool": step.tool_name,
                        "result": result.data,
                        "success": result.success,
                    }

        elif step.step_type == ReasoningStepType.ANALYSIS:
            # Perform analysis using LLM
            if self.orchestrator.llm_integration:
                analysis_prompt = f"""Analyze the following:

Description: {step.description}

Context: {chain.context}

Provide a detailed analysis."""

                analysis = await self.orchestrator.llm_integration.generate_response(
                    prompt=analysis_prompt, temperature=0.4, max_tokens=1500
                )

                return {"type": "analysis", "result": analysis, "success": True}

        elif step.step_type == ReasoningStepType.SYNTHESIS:
            # Synthesize results from previous steps
            previous_results = [
                chain.context.get(f"step_{dep_id}_result")
                for dep_id in step.dependencies
            ]

            if self.orchestrator.llm_integration:
                synthesis_prompt = f"""Synthesize the following information:

Goal: {chain.goal}

Previous Results:
{previous_results}

Provide a coherent synthesis."""

                synthesis = await self.orchestrator.llm_integration.generate_response(
                    prompt=synthesis_prompt, temperature=0.5, max_tokens=2000
                )

                return {"type": "synthesis", "result": synthesis, "success": True}

        # Default result
        return {
            "type": step.step_type.value,
            "result": f"Step {step.step_id} executed",
            "success": True,
        }

    async def _verify_reasoning_step(
        self, step: ReasoningStep, result: Dict[str, Any], chain: ReasoningChain
    ) -> TaskVerificationResult:
        """Verify a reasoning step result"""

        # Use adaptive engine for verification
        task = {
            "id": step.step_id,
            "type": step.step_type.value,
            "description": step.description,
        }

        return await self.adaptive_engine.verify_task_completion(task, result)

    async def _refine_reasoning_step(
        self,
        step: ReasoningStep,
        verification: TaskVerificationResult,
        chain: ReasoningChain,
    ) -> ReasoningStep:
        """Refine a reasoning step based on verification feedback"""

        refined_step = ReasoningStep(
            step_id=f"{step.step_id}_refined",
            step_type=step.step_type,
            description=f"{step.description} (Refined based on: {', '.join(verification.suggestions)})",
            tool_name=step.tool_name,
            parameters=step.parameters.copy(),
            dependencies=step.dependencies.copy(),
            success_criteria=step.success_criteria.copy(),
        )

        # Add refinement hints to parameters
        refined_step.parameters["_refinement_issues"] = verification.issues_found
        refined_step.parameters["_refinement_suggestions"] = verification.suggestions

        return refined_step


__all__ = [
    "MultiStepReasoningEngine",
    "ReasoningChain",
    "ReasoningStep",
    "ReasoningStepType",
]
