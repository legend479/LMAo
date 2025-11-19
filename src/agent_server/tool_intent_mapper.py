"""
Tool-Intent Mapping System
Maps user intents to appropriate tool combinations
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import re


class EnhancedIntentType(Enum):
    """Enhanced intent types with tool-specific categories"""

    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    CONTENT_GENERATION = "content_generation"
    DOCUMENT_EXPORT = "document_export"
    EMAIL_AUTOMATION = "email_automation"
    ANALYSIS = "analysis"
    MULTI_STEP = "multi_step"
    GENERAL_QUERY = "general_query"


@dataclass
class ToolCapability:
    """Describes what a tool can do"""

    tool_name: str
    primary_intents: List[EnhancedIntentType]
    secondary_intents: List[EnhancedIntentType]
    required_parameters: List[str]
    optional_parameters: List[str]
    output_type: str
    can_chain_with: List[str]  # Tools that can use this tool's output
    keywords: List[str]  # Keywords that suggest this tool


@dataclass
class IntentMatch:
    """Result of intent classification"""

    primary_intent: EnhancedIntentType
    secondary_intents: List[EnhancedIntentType]
    confidence: float
    suggested_tools: List[str]
    tool_sequence: List[Tuple[str, Dict[str, Any]]]  # (tool_name, parameters)
    reasoning: str


class ToolIntentMapper:
    """Maps intents to appropriate tools and creates execution sequences"""

    def __init__(self, llm_integration=None):
        self.tool_capabilities = self._initialize_tool_capabilities()
        self.intent_patterns = self._initialize_intent_patterns()
        self.tool_chains = self._initialize_tool_chains()
        self.llm_integration = llm_integration
        self._initialized = False

    def _initialize_tool_capabilities(self) -> Dict[str, ToolCapability]:
        """Define capabilities of each available tool"""
        return {
            "knowledge_retrieval": ToolCapability(
                tool_name="knowledge_retrieval",
                primary_intents=[EnhancedIntentType.KNOWLEDGE_RETRIEVAL],
                secondary_intents=[
                    EnhancedIntentType.CONTENT_GENERATION,
                    EnhancedIntentType.ANALYSIS,
                ],
                required_parameters=["query"],
                optional_parameters=["max_results", "filters"],
                output_type="text_chunks",
                can_chain_with=[
                    "content_generation",
                    "document_generation",
                    "email_automation",
                ],
                keywords=[
                    "explain",
                    "what is",
                    "how does",
                    "tell me about",
                    "find",
                    "search",
                ],
            ),
            "code_generation": ToolCapability(
                tool_name="code_generation",
                primary_intents=[EnhancedIntentType.CODE_GENERATION],
                secondary_intents=[],
                required_parameters=["description", "language"],
                optional_parameters=["requirements", "style_guide"],
                output_type="code",
                can_chain_with=["compiler_runtime", "document_generation"],
                keywords=[
                    "write code",
                    "implement",
                    "create function",
                    "build class",
                    "develop",
                ],
            ),
            "compiler_runtime": ToolCapability(
                tool_name="compiler_runtime",
                primary_intents=[EnhancedIntentType.CODE_EXECUTION],
                secondary_intents=[EnhancedIntentType.ANALYSIS],
                required_parameters=["code", "language"],
                optional_parameters=["test_cases", "timeout"],
                output_type="execution_result",
                can_chain_with=["document_generation", "email_automation"],
                keywords=["run", "execute", "test", "compile", "validate"],
            ),
            "content_generation": ToolCapability(
                tool_name="content_generation",
                primary_intents=[EnhancedIntentType.CONTENT_GENERATION],
                secondary_intents=[],
                required_parameters=["topic"],
                optional_parameters=["audience", "content_type", "length"],
                output_type="formatted_text",
                can_chain_with=["document_generation", "email_automation"],
                keywords=[
                    "generate content",
                    "create tutorial",
                    "write guide",
                    "explain",
                    "describe",
                ],
            ),
            "document_generation": ToolCapability(
                tool_name="document_generation",
                primary_intents=[EnhancedIntentType.DOCUMENT_EXPORT],
                secondary_intents=[],
                required_parameters=["content", "format"],
                optional_parameters=["template", "filename"],
                output_type="file_path",
                can_chain_with=["email_automation"],
                keywords=[
                    "create pdf",
                    "generate pdf",
                    "make pdf",
                    "create document",
                    "generate document",
                    "make document",
                    "create report",
                    "generate report",
                    "export to",
                    "save as",
                    "docx",
                    "pptx",
                    "pdf about",
                    "document about",
                    "report about",
                    "report on",
                ],
            ),
            "email_automation": ToolCapability(
                tool_name="email_automation",
                primary_intents=[EnhancedIntentType.EMAIL_AUTOMATION],
                secondary_intents=[],
                required_parameters=["recipients", "subject", "body"],
                optional_parameters=["attachments", "priority"],
                output_type="send_status",
                can_chain_with=[],
                keywords=["send email", "email to", "mail", "notify"],
            ),
        }

    def _initialize_intent_patterns(
        self,
    ) -> Dict[EnhancedIntentType, List[Dict[str, Any]]]:
        """Define patterns for intent detection"""
        return {
            # DOCUMENT_EXPORT should be checked with high priority
            EnhancedIntentType.DOCUMENT_EXPORT: [
                {
                    "pattern": r"\b(create|generate|make|export)\s+(pdf|docx|pptx|document|report)\b",
                    "weight": 1.0,
                },
                {
                    "pattern": r"\b(document|report|pdf|docx|pptx)\s+(about|on|for)\b",
                    "weight": 1.0,
                },
                {"pattern": r"\bsave\s+as\s+(pdf|docx|pptx)\b", "weight": 1.0},
                {"pattern": r"\bexport\s+to\b", "weight": 0.9},
                {
                    "pattern": r"\bgenerate\s+.*\s+(document|report|pdf|docx|pptx)\b",
                    "weight": 1.0,
                },
            ],
            EnhancedIntentType.EMAIL_AUTOMATION: [
                {"pattern": r"\b(send|email|mail)\s+(to|me|them)", "weight": 1.0},
                {"pattern": r"\bemail\s+.*\s+to\b", "weight": 1.0},
                {"pattern": r"\bnotify\s+.*\s+by\s+email\b", "weight": 0.9},
            ],
            EnhancedIntentType.CODE_EXECUTION: [
                {
                    "pattern": r"\b(run|execute|test|compile|validate)\s+(code|program|script)\b",
                    "weight": 1.0,
                },
                {"pattern": r"\brun\s+this\b", "weight": 0.9},
                {"pattern": r"\btest\s+the\s+code\b", "weight": 0.9},
            ],
            EnhancedIntentType.CODE_GENERATION: [
                {
                    "pattern": r"\b(write|create|implement|build|develop)\s+(code|function|class|program)\b",
                    "weight": 1.0,
                },
                {
                    "pattern": r"\bimplement\s+.*\s+in\s+(python|java|javascript)\b",
                    "weight": 1.0,
                },
                {"pattern": r"\bcode\s+for\b", "weight": 0.8},
            ],
            EnhancedIntentType.KNOWLEDGE_RETRIEVAL: [
                {
                    "pattern": r"\b(what|how|why|when|where)\s+(is|are|does|do)\b",
                    "weight": 0.9,
                },
                {"pattern": r"\b(explain|describe|tell\s+me\s+about)\b", "weight": 0.9},
                {"pattern": r"\bfind\s+information\b", "weight": 0.8},
            ],
            EnhancedIntentType.CONTENT_GENERATION: [
                {
                    "pattern": r"\b(generate|create|write)\s+(content|tutorial|guide|article|blog)\b",
                    "weight": 1.0,
                },
                {
                    "pattern": r"\bcreate\s+a\s+(tutorial|guide|explanation|article)\b",
                    "weight": 0.9,
                },
            ],
            EnhancedIntentType.MULTI_STEP: [
                {
                    "pattern": r"\b(and\s+then|after\s+that|next|also|additionally)\b",
                    "weight": 0.8,
                },
                {"pattern": r"\b(first|second|third|finally)\b", "weight": 0.7},
            ],
        }

    def _initialize_tool_chains(self) -> Dict[str, List[str]]:
        """Define common tool execution chains"""
        return {
            "code_with_execution": ["code_generation", "compiler_runtime"],
            "content_to_document": ["content_generation", "document_generation"],
            "content_to_email": ["content_generation", "email_automation"],
            "document_to_email": [
                "content_generation",
                "document_generation",
                "email_automation",
            ],
            "knowledge_to_content": ["knowledge_retrieval", "content_generation"],
            "knowledge_to_document": [
                "knowledge_retrieval",
                "content_generation",
                "document_generation",
            ],
        }

    async def initialize(self, llm_integration=None):
        """Initialize the mapper with LLM integration"""
        if llm_integration:
            self.llm_integration = llm_integration
        self._initialized = True

    async def classify_intent_async(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> IntentMatch:
        """Async version with LLM-based classification (primary) and pattern fallback"""
        from src.shared.logging import get_logger

        logger = get_logger(__name__)

        # Try LLM-based classification first for better natural language understanding
        if self.llm_integration:
            try:
                llm_result = await self._classify_with_llm(message, context)
                if llm_result:
                    logger.info(
                        "Using LLM-based intent classification",
                        intent=llm_result.primary_intent.value,
                        confidence=llm_result.confidence,
                    )
                    return llm_result
            except Exception as e:
                logger.warning(
                    f"LLM intent classification failed, falling back to patterns: {str(e)}"
                )

        # Fallback to pattern-based classification
        logger.info("Using pattern-based intent classification (fallback)")
        return self.classify_intent(message, context)

    async def _classify_with_llm(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[IntentMatch]:
        """Use LLM for natural language understanding of intent"""
        from src.shared.logging import get_logger

        logger = get_logger(__name__)

        if not self.llm_integration:
            return None

        # Build comprehensive system prompt with tool information
        tool_descriptions = []
        for tool_name, capability in self.tool_capabilities.items():
            tool_descriptions.append(
                f"- {tool_name}: {', '.join(capability.keywords[:3])}"
            )

        system_prompt = f"""You are an expert at classifying user intents for an AI agent with multiple tools.

Available intent categories and their purposes:
- DOCUMENT_EXPORT: User wants to create/generate/export documents (PDF, DOCX, PPT, reports)
- EMAIL_AUTOMATION: User wants to send emails or notifications
- CODE_GENERATION: User wants to write, implement, or create code
- CODE_EXECUTION: User wants to run, test, or execute code
- CONTENT_GENERATION: User wants to create tutorials, guides, articles, or explanatory content
- KNOWLEDGE_RETRIEVAL: User wants to learn, understand, or get information about something
- ANALYSIS: User wants to analyze, review, evaluate, or compare something
- MULTI_STEP: User's request involves multiple sequential tasks
- GENERAL_QUERY: Simple questions or general assistance

Available tools:
{chr(10).join(tool_descriptions)}

IMPORTANT DISTINCTIONS:
- "generate a document/report/PDF about X" → DOCUMENT_EXPORT (creates a file)
- "write a tutorial/guide about X" → CONTENT_GENERATION (creates text content)
- "explain what is X" → KNOWLEDGE_RETRIEVAL (retrieves information)

Respond with a JSON object containing:
{{
  "primary_intent": "intent_name",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "suggested_tools": ["tool1", "tool2"]
}}"""

        # Add context if available
        context_info = ""
        if context and context.get("history"):
            recent = (
                context["history"][-2:]
                if len(context["history"]) > 2
                else context["history"]
            )
            if recent:
                context_info = f"\n\nRecent conversation context:\n"
                for msg in recent:
                    content = msg.get("content", "")[:100]
                    context_info += f"- {content}\n"

        prompt = f"{context_info}\nUser message: {message}\n\nClassify this intent and suggest appropriate tools:"

        try:
            response = await self.llm_integration.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200,
            )

            # Parse LLM response
            import json

            # Try to extract JSON from response
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = (
                    response_clean.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0].strip()

            result = json.loads(response_clean)

            # Map intent name to EnhancedIntentType
            intent_name = result.get("primary_intent", "").upper()
            primary_intent = None

            for intent in EnhancedIntentType:
                if intent.value.upper() == intent_name or intent.name == intent_name:
                    primary_intent = intent
                    break

            if not primary_intent:
                logger.warning(f"LLM returned unknown intent: {intent_name}")
                return None

            confidence = float(result.get("confidence", 0.7))
            reasoning = result.get("reasoning", "LLM-based classification")
            suggested_tools = result.get("suggested_tools", [])

            # ALWAYS use _create_tool_sequence to build proper tool chains
            # This ensures document generation gets content_generation first, etc.
            final_suggested_tools, tool_sequence, chain_reasoning = (
                self._create_tool_sequence(primary_intent, [], message, context)
            )

            # Combine LLM reasoning with chain reasoning
            combined_reasoning = (
                f"{reasoning}; {chain_reasoning}" if chain_reasoning else reasoning
            )

            logger.info(
                "LLM classification successful",
                intent=primary_intent.value,
                confidence=confidence,
                tools=suggested_tools,
                tool_sequence=[t for t, _ in tool_sequence],
                chain_reasoning=chain_reasoning,
            )

            return IntentMatch(
                primary_intent=primary_intent,
                secondary_intents=[],
                confidence=confidence,
                suggested_tools=final_suggested_tools,
                tool_sequence=tool_sequence,
                reasoning=combined_reasoning,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"LLM response was: {response[:200]}")
            return None
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return None

    def classify_intent(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> IntentMatch:
        """Synchronous pattern-based classification (fallback method)"""

        message_lower = message.lower()
        intent_scores = {intent: 0.0 for intent in EnhancedIntentType}

        # Pattern-based scoring
        matched_patterns = []
        for intent, patterns in self.intent_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], message_lower):
                    intent_scores[intent] += pattern_info["weight"]
                    matched_patterns.append((intent.value, pattern_info["pattern"]))

        # Keyword-based scoring
        matched_keywords = []
        for tool_name, capability in self.tool_capabilities.items():
            for keyword in capability.keywords:
                if keyword in message_lower:
                    matched_keywords.append((tool_name, keyword))
                    for intent in capability.primary_intents:
                        intent_scores[intent] += 0.5
                    for intent in capability.secondary_intents:
                        intent_scores[intent] += 0.2

        # Determine primary and secondary intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)

        primary_intent = (
            sorted_intents[0][0]
            if sorted_intents[0][1] > 0
            else EnhancedIntentType.GENERAL_QUERY
        )
        primary_score = sorted_intents[0][1]

        secondary_intents = [
            intent
            for intent, score in sorted_intents[1:4]
            if score > 0.3 and score >= primary_score * 0.5
        ]

        # Calculate confidence
        confidence = min(primary_score / 2.0, 1.0) if primary_score > 0 else 0.3

        # Log classification details for debugging
        from src.shared.logging import get_logger

        logger = get_logger(__name__)
        logger.debug(
            "Pattern-based intent classification",
            message=message[:100],
            primary_intent=primary_intent.value,
            primary_score=primary_score,
            confidence=confidence,
            matched_patterns=matched_patterns[:5],  # Top 5
            matched_keywords=matched_keywords[:5],  # Top 5
            top_intents=[(intent.value, score) for intent, score in sorted_intents[:3]],
        )

        # Suggest tools and create sequence
        suggested_tools, tool_sequence, reasoning = self._create_tool_sequence(
            primary_intent, secondary_intents, message, context
        )

        return IntentMatch(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            suggested_tools=suggested_tools,
            tool_sequence=tool_sequence,
            reasoning=reasoning,
        )

    def _create_tool_sequence(
        self,
        primary_intent: EnhancedIntentType,
        secondary_intents: List[EnhancedIntentType],
        message: str,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[Tuple[str, Dict[str, Any]]], str]:
        """Create optimal tool execution sequence"""

        suggested_tools = []
        tool_sequence = []
        reasoning_parts = []

        # Find tools for primary intent
        primary_tools = [
            tool_name
            for tool_name, cap in self.tool_capabilities.items()
            if primary_intent in cap.primary_intents
        ]

        if primary_tools:
            suggested_tools.extend(primary_tools)
            reasoning_parts.append(
                f"Primary intent '{primary_intent.value}' maps to {', '.join(primary_tools)}"
            )

        # Check for tool chains
        message_lower = message.lower()

        # Document export chain
        if primary_intent == EnhancedIntentType.DOCUMENT_EXPORT:
            if "content" not in message_lower and "text" not in message_lower:
                # Need to generate content first
                tool_sequence.append(("content_generation", {"for_document": True}))
                tool_sequence.append(
                    ("document_generation", {"content_from_previous": True})
                )
                reasoning_parts.append(
                    "Document export requires content generation first"
                )
            else:
                tool_sequence.append(("document_generation", {}))

        # Email with attachment chain
        elif primary_intent == EnhancedIntentType.EMAIL_AUTOMATION:
            if any(word in message_lower for word in ["attach", "pdf", "document"]):
                tool_sequence.append(("content_generation", {}))
                tool_sequence.append(("document_generation", {}))
                tool_sequence.append(("email_automation", {"attach_previous": True}))
                reasoning_parts.append(
                    "Email with attachment requires document generation"
                )
            else:
                tool_sequence.append(("email_automation", {}))

        # Code execution chain
        elif primary_intent == EnhancedIntentType.CODE_EXECUTION:
            if "write" in message_lower or "create" in message_lower:
                tool_sequence.append(("code_generation", {}))
                tool_sequence.append(("compiler_runtime", {"code_from_previous": True}))
                reasoning_parts.append("Code execution requires generation first")
            else:
                tool_sequence.append(("compiler_runtime", {}))

        # Content generation chain
        elif primary_intent == EnhancedIntentType.CONTENT_GENERATION:
            if any(word in message_lower for word in ["research", "find", "about"]):
                tool_sequence.append(("knowledge_retrieval", {}))
                tool_sequence.append(
                    ("content_generation", {"context_from_previous": True})
                )
                reasoning_parts.append(
                    "Content generation benefits from knowledge retrieval"
                )
            else:
                tool_sequence.append(("content_generation", {}))

        # Knowledge retrieval
        elif primary_intent == EnhancedIntentType.KNOWLEDGE_RETRIEVAL:
            tool_sequence.append(("knowledge_retrieval", {}))

        # Default to general processing
        else:
            tool_sequence.append(("general_processing", {}))
            reasoning_parts.append("Using general processing for query")

        # Extract unique tools from sequence
        if not suggested_tools:
            suggested_tools = list(set(tool for tool, _ in tool_sequence))

        reasoning = (
            "; ".join(reasoning_parts)
            if reasoning_parts
            else "Standard single-tool execution"
        )

        return suggested_tools, tool_sequence, reasoning

    def get_tool_parameters(
        self, tool_name: str, message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract parameters for a specific tool from message and context"""

        parameters = {}
        message_lower = message.lower()

        if tool_name == "email_automation":
            # Extract email parameters
            email_match = re.search(
                r"(?:email|send|mail)\s+(?:to\s+)?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                message,
            )
            if email_match:
                parameters["recipients"] = [email_match.group(1)]

            # Extract subject
            subject_match = re.search(
                r'subject[:\s]+["\']?([^"\']+)["\']?', message, re.IGNORECASE
            )
            if subject_match:
                parameters["subject"] = subject_match.group(1).strip()

        elif tool_name == "document_generation":
            # Extract format
            for fmt in ["pdf", "docx", "pptx"]:
                if fmt in message_lower:
                    parameters["format"] = fmt
                    break

            # Extract filename
            filename_match = re.search(
                r'(?:save|name|call)\s+(?:as|it)\s+["\']?([^"\']+)["\']?',
                message,
                re.IGNORECASE,
            )
            if filename_match:
                parameters["filename"] = filename_match.group(1).strip()

        elif tool_name == "code_generation":
            # Extract language
            languages = [
                "python",
                "javascript",
                "java",
                "c++",
                "go",
                "rust",
                "typescript",
            ]
            for lang in languages:
                if lang in message_lower:
                    parameters["language"] = lang
                    break

        elif tool_name == "compiler_runtime":
            # Extract language
            languages = ["python", "javascript", "java", "c++", "go", "rust"]
            for lang in languages:
                if lang in message_lower:
                    parameters["language"] = lang
                    break

        return parameters
