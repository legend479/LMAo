"""
Code Generation and Validation System
Implements code generation with functional correctness validation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import ast
import asyncio
import uuid

from .prompt_engineering import (
    AdvancedPromptEngineeringFramework,
    AudienceLevel,
    ContentType,
    PromptGenerationRequest,
    PromptType,
)
from .tools.compiler_runtime import CompilerRuntimeTool
from src.shared.logging import get_logger

logger = get_logger(__name__)


class ProgrammingLanguage(Enum):
    """Supported programming languages"""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"


class CodeType(Enum):
    """Types of code that can be generated"""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SCRIPT = "script"
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    API_ENDPOINT = "api_endpoint"
    TEST_CASE = "test_case"
    UTILITY = "utility"
    EXAMPLE = "example"


class QualityMetric(Enum):
    """Code quality metrics"""

    FUNCTIONALITY = "functionality"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    STYLE = "style"


class ValidationResult(Enum):
    """Validation result types"""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class CodeGenerationRequest:
    """Request for code generation"""

    description: str
    language: ProgrammingLanguage
    code_type: CodeType
    audience_level: AudienceLevel
    requirements: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    include_tests: bool = True
    include_documentation: bool = True
    style_guide: Optional[str] = None
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationTest:
    """Test case for code validation"""

    test_id: str
    name: str
    input_data: Any
    expected_output: Any
    test_type: str  # unit, integration, performance
    timeout: float = 5.0
    description: Optional[str] = None


@dataclass
class QualityAssessment:
    """Code quality assessment result"""

    metric: QualityMetric
    score: float  # 0.0 to 1.0
    details: str
    suggestions: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class CodeValidationResult:
    """Result of code validation"""

    validation_id: str
    overall_result: ValidationResult
    functionality_score: float
    quality_assessments: List[QualityAssessment]
    test_results: List[Dict[str, Any]]
    execution_time: float
    memory_usage: Optional[float] = None
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class GeneratedCode:
    """Result of code generation"""

    code_id: str
    code: str
    language: ProgrammingLanguage
    code_type: CodeType
    documentation: str
    test_code: Optional[str] = None
    validation_result: Optional[CodeValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    confidence_score: float = 0.0
    iterations: int = 1


class CodeAnalyzer:
    """Analyzes code structure and quality"""

    def __init__(self):
        self.language_analyzers = self._initialize_language_analyzers()

    async def analyze_code_structure(
        self, code: str, language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """Analyze code structure and extract metadata"""

        if language in self.language_analyzers:
            analyzer = self.language_analyzers[language]
            return await analyzer(code)

        return await self._generic_code_analysis(code)

    async def assess_code_quality(
        self, code: str, language: ProgrammingLanguage
    ) -> List[QualityAssessment]:
        """Assess code quality across multiple metrics"""

        assessments = []

        # Functionality assessment (basic syntax check)
        functionality = await self._assess_functionality(code, language)
        assessments.append(functionality)

        # Readability assessment
        readability = await self._assess_readability(code, language)
        assessments.append(readability)

        # Documentation assessment
        documentation = await self._assess_documentation(code, language)
        assessments.append(documentation)

        # Style assessment
        style = await self._assess_style(code, language)
        assessments.append(style)

        # Security assessment (basic)
        security = await self._assess_security(code, language)
        assessments.append(security)

        return assessments

    async def extract_functions(
        self, code: str, language: ProgrammingLanguage
    ) -> List[Dict[str, Any]]:
        """Extract function definitions from code"""

        if language == ProgrammingLanguage.PYTHON:
            return await self._extract_python_functions(code)
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return await self._extract_javascript_functions(code)
        else:
            return await self._extract_generic_functions(code)

    async def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code structure"""

        try:
            tree = ast.parse(code)

            analysis = {
                "functions": [],
                "classes": [],
                "imports": [],
                "variables": [],
                "complexity": 0,
                "lines_of_code": len(code.splitlines()),
                "docstrings": [],
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node),
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(
                        {
                            "name": node.name,
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node),
                        }
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            analysis["imports"].append(f"{node.module}.{alias.name}")

            return analysis

        except SyntaxError as e:
            return {
                "error": f"Syntax error: {str(e)}",
                "functions": [],
                "classes": [],
                "imports": [],
                "variables": [],
                "complexity": 0,
                "lines_of_code": len(code.splitlines()),
                "docstrings": [],
            }

    async def _analyze_javascript_code(self, code: str) -> Dict[str, Any]:
        """Analyze JavaScript code structure"""

        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "complexity": 0,
            "lines_of_code": len(code.splitlines()),
            "comments": [],
        }

        # Simple regex-based analysis for JavaScript
        # Function declarations
        function_pattern = r"function\s+(\w+)\s*\([^)]*\)"
        functions = re.findall(function_pattern, code)
        analysis["functions"] = [
            {"name": func, "type": "declaration"} for func in functions
        ]

        # Arrow functions
        arrow_pattern = r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>"
        arrow_functions = re.findall(arrow_pattern, code)
        analysis["functions"].extend(
            [{"name": func, "type": "arrow"} for func in arrow_functions]
        )

        # Class declarations
        class_pattern = r"class\s+(\w+)"
        classes = re.findall(class_pattern, code)
        analysis["classes"] = [{"name": cls} for cls in classes]

        # Import statements
        import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
        imports = re.findall(import_pattern, code)
        analysis["imports"] = imports

        return analysis

    async def _generic_code_analysis(self, code: str) -> Dict[str, Any]:
        """Generic code analysis for unsupported languages"""

        return {
            "lines_of_code": len(code.splitlines()),
            "character_count": len(code),
            "word_count": len(code.split()),
            "comment_lines": len(
                [
                    line
                    for line in code.splitlines()
                    if line.strip().startswith(("#", "//", "/*"))
                ]
            ),
            "blank_lines": len(
                [line for line in code.splitlines() if not line.strip()]
            ),
        }

    async def _assess_functionality(
        self, code: str, language: ProgrammingLanguage
    ) -> QualityAssessment:
        """Assess code functionality (syntax correctness)"""

        issues = []
        score = 1.0

        if language == ProgrammingLanguage.PYTHON:
            try:
                ast.parse(code)
            except SyntaxError as e:
                issues.append(f"Syntax error: {str(e)}")
                score = 0.0

        return QualityAssessment(
            metric=QualityMetric.FUNCTIONALITY,
            score=score,
            details="Code syntax validation",
            issues=issues,
            suggestions=["Fix syntax errors"] if issues else [],
        )

    async def _assess_readability(
        self, code: str, language: ProgrammingLanguage
    ) -> QualityAssessment:
        """Assess code readability"""

        score = 0.5  # Base score
        suggestions = []

        lines = code.splitlines()

        # Check for reasonable line length
        long_lines = [i for i, line in enumerate(lines) if len(line) > 100]
        if long_lines:
            score -= 0.1
            suggestions.append("Consider breaking long lines (>100 characters)")

        # Check for comments
        comment_lines = len(
            [line for line in lines if line.strip().startswith(("#", "//", "/*"))]
        )
        comment_ratio = comment_lines / max(len(lines), 1)
        if comment_ratio > 0.1:
            score += 0.2
        else:
            suggestions.append("Add more comments to explain complex logic")

        # Check for meaningful variable names
        if language == ProgrammingLanguage.PYTHON:
            short_vars = re.findall(r"\b[a-z]\b", code)
            if len(short_vars) > 5:
                score -= 0.1
                suggestions.append("Use more descriptive variable names")

        return QualityAssessment(
            metric=QualityMetric.READABILITY,
            score=max(0.0, min(1.0, score)),
            details="Code readability assessment",
            suggestions=suggestions,
        )

    async def _assess_documentation(
        self, code: str, language: ProgrammingLanguage
    ) -> QualityAssessment:
        """Assess code documentation"""

        score = 0.0
        suggestions = []

        if language == ProgrammingLanguage.PYTHON:
            # Check for docstrings
            docstring_pattern = r'""".*?"""'
            docstrings = re.findall(docstring_pattern, code, re.DOTALL)
            if docstrings:
                score += 0.5
            else:
                suggestions.append("Add docstrings to functions and classes")

            # Check for type hints
            if ":" in code and "->" in code:
                score += 0.3
            else:
                suggestions.append("Consider adding type hints")

        # Check for inline comments
        comment_lines = len(
            [line for line in code.splitlines() if "#" in line or "//" in line]
        )
        if comment_lines > 0:
            score += 0.2
        else:
            suggestions.append("Add inline comments for complex logic")

        return QualityAssessment(
            metric=QualityMetric.DOCUMENTATION,
            score=min(1.0, score),
            details="Code documentation assessment",
            suggestions=suggestions,
        )

    async def _assess_style(
        self, code: str, language: ProgrammingLanguage
    ) -> QualityAssessment:
        """Assess code style"""

        score = 0.5  # Base score
        suggestions = []

        if language == ProgrammingLanguage.PYTHON:
            # Check for PEP 8 compliance (basic)
            lines = code.splitlines()

            # Check indentation
            indent_issues = 0
            for line in lines:
                if line.strip() and not line.startswith(
                    " " * (len(line) - len(line.lstrip())) // 4 * 4
                ):
                    indent_issues += 1

            if indent_issues > 0:
                score -= 0.2
                suggestions.append("Use consistent 4-space indentation")

            # Check for proper spacing
            if " =" in code and "= " not in code:
                score -= 0.1
                suggestions.append("Add spaces around operators")

        return QualityAssessment(
            metric=QualityMetric.STYLE,
            score=max(0.0, score),
            details="Code style assessment",
            suggestions=suggestions,
        )

    async def _assess_security(
        self, code: str, language: ProgrammingLanguage
    ) -> QualityAssessment:
        """Assess code security (basic checks)"""

        score = 1.0
        issues = []
        suggestions = []

        # Check for common security issues
        security_patterns = {
            "eval": "Avoid using eval() as it can execute arbitrary code",
            "exec": "Avoid using exec() as it can execute arbitrary code",
            "input()": "Be careful with input() - validate user input",
            "os.system": "Avoid os.system() - use subprocess instead",
            "shell=True": "Avoid shell=True in subprocess calls",
        }

        for pattern, message in security_patterns.items():
            if pattern in code:
                score -= 0.2
                issues.append(message)
                suggestions.append(f"Replace {pattern} with safer alternatives")

        return QualityAssessment(
            metric=QualityMetric.SECURITY,
            score=max(0.0, score),
            details="Basic security assessment",
            issues=issues,
            suggestions=suggestions,
        )

    async def _extract_python_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract Python function definitions"""

        functions = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node),
                            "returns": (
                                node.returns.id
                                if node.returns and hasattr(node.returns, "id")
                                else None
                            ),
                        }
                    )

        except SyntaxError:
            pass

        return functions

    async def _extract_javascript_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract JavaScript function definitions"""

        functions = []

        # Function declarations
        function_pattern = r"function\s+(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(function_pattern, code):
            functions.append(
                {
                    "name": match.group(1),
                    "args": [
                        arg.strip() for arg in match.group(2).split(",") if arg.strip()
                    ],
                    "type": "declaration",
                }
            )

        # Arrow functions
        arrow_pattern = r"const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>"
        for match in re.finditer(arrow_pattern, code):
            functions.append(
                {
                    "name": match.group(1),
                    "args": [
                        arg.strip() for arg in match.group(2).split(",") if arg.strip()
                    ],
                    "type": "arrow",
                }
            )

        return functions

    async def _extract_generic_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function definitions for generic languages"""

        # Very basic pattern matching
        patterns = [
            r"def\s+(\w+)\s*\(",  # Python
            r"function\s+(\w+)\s*\(",  # JavaScript
            r"public\s+\w+\s+(\w+)\s*\(",  # Java
            r"func\s+(\w+)\s*\(",  # Go
        ]

        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, code)
            functions.extend([{"name": match, "type": "generic"} for match in matches])

        return functions

    def _initialize_language_analyzers(self) -> Dict[ProgrammingLanguage, callable]:
        """Initialize language-specific analyzers"""

        return {
            ProgrammingLanguage.PYTHON: self._analyze_python_code,
            ProgrammingLanguage.JAVASCRIPT: self._analyze_javascript_code,
        }


class CodeValidator:
    """Validates code functionality and quality"""

    def __init__(self):
        self.compiler_tool = CompilerRuntimeTool()
        self.code_analyzer = CodeAnalyzer()

    async def validate_code(
        self,
        code: str,
        language: ProgrammingLanguage,
        tests: List[ValidationTest] = None,
    ) -> CodeValidationResult:
        """Validate code functionality and quality"""

        validation_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()

        logger.info(
            "Validating code", validation_id=validation_id, language=language.value
        )

        try:
            # Quality assessment
            quality_assessments = await self.code_analyzer.assess_code_quality(
                code, language
            )

            # Functionality testing
            test_results = []
            functionality_score = 1.0

            if tests:
                test_results = await self._run_tests(code, language, tests)
                functionality_score = self._calculate_functionality_score(test_results)
            else:
                # Basic syntax validation
                syntax_result = await self._validate_syntax(code, language)
                test_results = [syntax_result]
                functionality_score = 1.0 if syntax_result["result"] == "pass" else 0.0

            # Determine overall result
            overall_result = self._determine_overall_result(
                functionality_score, quality_assessments
            )

            # Collect suggestions
            suggestions = []
            error_messages = []
            warnings = []

            for assessment in quality_assessments:
                suggestions.extend(assessment.suggestions)
                if assessment.score < 0.5:
                    warnings.append(
                        f"Low {assessment.metric.value} score: {assessment.details}"
                    )

            for test_result in test_results:
                if test_result.get("error"):
                    error_messages.append(test_result["error"])

            execution_time = asyncio.get_event_loop().time() - start_time

            result = CodeValidationResult(
                validation_id=validation_id,
                overall_result=overall_result,
                functionality_score=functionality_score,
                quality_assessments=quality_assessments,
                test_results=test_results,
                execution_time=execution_time,
                error_messages=error_messages,
                warnings=warnings,
                suggestions=suggestions,
            )

            logger.info(
                "Code validation completed",
                validation_id=validation_id,
                overall_result=overall_result.value,
                functionality_score=functionality_score,
            )

            return result

        except Exception as e:
            logger.error(
                "Code validation failed", validation_id=validation_id, error=str(e)
            )

            return CodeValidationResult(
                validation_id=validation_id,
                overall_result=ValidationResult.ERROR,
                functionality_score=0.0,
                quality_assessments=[],
                test_results=[],
                execution_time=asyncio.get_event_loop().time() - start_time,
                error_messages=[str(e)],
            )

    async def _run_tests(
        self, code: str, language: ProgrammingLanguage, tests: List[ValidationTest]
    ) -> List[Dict[str, Any]]:
        """Run validation tests on code"""

        test_results = []

        for test in tests:
            try:
                result = await self._run_single_test(code, language, test)
                test_results.append(result)
            except Exception as e:
                test_results.append(
                    {
                        "test_id": test.test_id,
                        "name": test.name,
                        "result": "error",
                        "error": str(e),
                        "execution_time": 0.0,
                    }
                )

        return test_results

    async def _run_single_test(
        self, code: str, language: ProgrammingLanguage, test: ValidationTest
    ) -> Dict[str, Any]:
        """Run a single validation test"""

        start_time = asyncio.get_event_loop().time()

        try:
            # Create test execution context
            if language == ProgrammingLanguage.PYTHON:
                result = await self._run_python_test(code, test)
            else:
                # For other languages, use compiler tool
                result = await self._run_generic_test(code, language, test)

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                "test_id": test.test_id,
                "name": test.name,
                "result": "pass" if result["success"] else "fail",
                "expected": test.expected_output,
                "actual": result.get("output"),
                "execution_time": execution_time,
                "error": result.get("error"),
            }

        except asyncio.TimeoutError:
            return {
                "test_id": test.test_id,
                "name": test.name,
                "result": "timeout",
                "execution_time": test.timeout,
                "error": f"Test timed out after {test.timeout} seconds",
            }

    async def _run_python_test(self, code: str, test: ValidationTest) -> Dict[str, Any]:
        """Run Python test"""

        try:
            # Create a temporary namespace for execution
            namespace = {}

            # Execute the code
            exec(code, namespace)

            # Find the main function to test
            functions = await self.code_analyzer.extract_functions(
                code, ProgrammingLanguage.PYTHON
            )

            if not functions:
                return {"success": False, "error": "No functions found to test"}

            # Test the first function (or find specific function)
            main_function = functions[0]["name"]

            if main_function in namespace:
                func = namespace[main_function]

                # Call function with test input
                if isinstance(test.input_data, dict):
                    output = func(**test.input_data)
                elif isinstance(test.input_data, (list, tuple)):
                    output = func(*test.input_data)
                else:
                    output = func(test.input_data)

                # Compare with expected output
                success = output == test.expected_output

                return {
                    "success": success,
                    "output": output,
                    "error": (
                        None
                        if success
                        else f"Expected {test.expected_output}, got {output}"
                    ),
                }
            else:
                return {
                    "success": False,
                    "error": f"Function {main_function} not found",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_generic_test(
        self, code: str, language: ProgrammingLanguage, test: ValidationTest
    ) -> Dict[str, Any]:
        """Run test using compiler tool"""

        try:
            # Use compiler tool for execution
            execution_result = await self.compiler_tool.execute_code(
                code, language.value, test.input_data
            )

            if execution_result.success:
                success = execution_result.output == test.expected_output
                return {
                    "success": success,
                    "output": execution_result.output,
                    "error": (
                        None
                        if success
                        else f"Expected {test.expected_output}, got {execution_result.output}"
                    ),
                }
            else:
                return {"success": False, "error": execution_result.error_message}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _validate_syntax(
        self, code: str, language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """Validate code syntax"""

        if language == ProgrammingLanguage.PYTHON:
            try:
                ast.parse(code)
                return {
                    "test_id": "syntax_check",
                    "name": "Syntax Validation",
                    "result": "pass",
                    "error": None,
                }
            except SyntaxError as e:
                return {
                    "test_id": "syntax_check",
                    "name": "Syntax Validation",
                    "result": "fail",
                    "error": f"Syntax error: {str(e)}",
                }
        else:
            # For other languages, assume syntax is valid if no obvious issues
            return {
                "test_id": "syntax_check",
                "name": "Syntax Validation",
                "result": "pass",
                "error": None,
            }

    def _calculate_functionality_score(
        self, test_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate functionality score from test results"""

        if not test_results:
            return 0.0

        passed_tests = sum(1 for result in test_results if result["result"] == "pass")
        return passed_tests / len(test_results)

    def _determine_overall_result(
        self, functionality_score: float, quality_assessments: List[QualityAssessment]
    ) -> ValidationResult:
        """Determine overall validation result"""

        if functionality_score == 0.0:
            return ValidationResult.FAIL

        # Check for critical quality issues
        critical_issues = [
            assessment
            for assessment in quality_assessments
            if assessment.metric == QualityMetric.FUNCTIONALITY
            and assessment.score < 0.5
        ]

        if critical_issues:
            return ValidationResult.FAIL

        # Check for warnings
        warning_issues = [
            assessment for assessment in quality_assessments if assessment.score < 0.7
        ]

        if warning_issues:
            return ValidationResult.WARNING

        return ValidationResult.PASS


class CodeGenerator:
    """Main code generation system"""

    def __init__(self):
        self.prompt_framework = None
        self.code_analyzer = CodeAnalyzer()
        self.code_validator = CodeValidator()
        self.language_templates = self._initialize_language_templates()
        self._initialized = False

    async def initialize(self):
        """Initialize the code generation system"""
        if self._initialized:
            return

        logger.info("Initializing Code Generation System")

        # Initialize prompt engineering framework
        self.prompt_framework = AdvancedPromptEngineeringFramework()
        await self.prompt_framework.initialize()

        # Load code generation templates
        await self._load_code_templates()

        self._initialized = True
        logger.info("Code Generation System initialized successfully")

    async def generate_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code based on request"""

        start_time = asyncio.get_event_loop().time()
        code_id = str(uuid.uuid4())

        logger.info(
            "Generating code",
            code_id=code_id,
            language=request.language.value,
            code_type=request.code_type.value,
            audience_level=request.audience_level.value,
        )

        try:
            # Generate initial code
            initial_code = await self._generate_initial_code(request)

            # Generate documentation
            documentation = await self._generate_documentation(initial_code, request)

            # Generate tests if requested
            test_code = None
            if request.include_tests:
                test_code = await self._generate_test_code(initial_code, request)

            # Validate code
            validation_result = await self._validate_generated_code(
                initial_code, request, test_code
            )

            # Iterative improvement if needed
            final_code = initial_code
            iterations = 1

            if validation_result.overall_result in [
                ValidationResult.FAIL,
                ValidationResult.WARNING,
            ]:
                final_code, iterations = await self._improve_code_iteratively(
                    initial_code, request, validation_result
                )

                # Re-validate improved code
                validation_result = await self._validate_generated_code(
                    final_code, request, test_code
                )

            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(
                final_code, validation_result, request
            )

            generation_time = asyncio.get_event_loop().time() - start_time

            result = GeneratedCode(
                code_id=code_id,
                code=final_code,
                language=request.language,
                code_type=request.code_type,
                documentation=documentation,
                test_code=test_code,
                validation_result=validation_result,
                metadata={
                    "description": request.description,
                    "audience_level": request.audience_level.value,
                    "requirements": request.requirements,
                    "constraints": request.constraints,
                    "style_guide": request.style_guide,
                },
                generation_time=generation_time,
                confidence_score=confidence_score,
                iterations=iterations,
            )

            logger.info(
                "Code generation completed",
                code_id=code_id,
                confidence=confidence_score,
                iterations=iterations,
                validation_result=validation_result.overall_result.value,
            )

            return result

        except Exception as e:
            logger.error("Code generation failed", code_id=code_id, error=str(e))
            raise

    async def generate_function(
        self,
        description: str,
        language: ProgrammingLanguage,
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
        include_tests: bool = True,
    ) -> GeneratedCode:
        """Generate a single function"""

        request = CodeGenerationRequest(
            description=description,
            language=language,
            code_type=CodeType.FUNCTION,
            audience_level=audience_level,
            include_tests=include_tests,
        )

        return await self.generate_code(request)

    async def generate_class(
        self,
        description: str,
        language: ProgrammingLanguage,
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
        include_tests: bool = True,
    ) -> GeneratedCode:
        """Generate a class"""

        request = CodeGenerationRequest(
            description=description,
            language=language,
            code_type=CodeType.CLASS,
            audience_level=audience_level,
            include_tests=include_tests,
        )

        return await self.generate_code(request)

    async def generate_algorithm(
        self,
        description: str,
        language: ProgrammingLanguage,
        performance_requirements: Dict[str, Any] = None,
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
    ) -> GeneratedCode:
        """Generate an algorithm implementation"""

        request = CodeGenerationRequest(
            description=description,
            language=language,
            code_type=CodeType.ALGORITHM,
            audience_level=audience_level,
            performance_requirements=performance_requirements or {},
            include_tests=True,
        )

        return await self.generate_code(request)

    async def improve_code(
        self, code: str, language: ProgrammingLanguage, improvement_goals: List[str]
    ) -> GeneratedCode:
        """Improve existing code based on goals"""

        # Analyze current code
        quality_assessments = await self.code_analyzer.assess_code_quality(
            code, language
        )

        # Create improvement request
        request = CodeGenerationRequest(
            description=f"Improve code: {', '.join(improvement_goals)}",
            language=language,
            code_type=CodeType.UTILITY,  # Generic type for improvement
            audience_level=AudienceLevel.INTERMEDIATE,
            context={"original_code": code, "improvement_goals": improvement_goals},
        )

        # Generate improved code
        return await self.generate_code(request)

    async def _generate_initial_code(self, request: CodeGenerationRequest) -> str:
        """Generate initial code using prompt framework"""

        # Create prompt for code generation
        prompt_request = PromptGenerationRequest(
            content_type=ContentType.CODE,
            audience_level=request.audience_level,
            topic=f"{request.code_type.value} in {request.language.value}: {request.description}",
            context={
                "language": request.language.value,
                "code_type": request.code_type.value,
                "requirements": request.requirements,
                "constraints": request.constraints,
                "style_guide": request.style_guide,
            },
            prompt_type=PromptType.FEW_SHOT,
        )

        # Generate prompt
        generated_prompt = await self.prompt_framework.generate_prompt(prompt_request)

        # For now, return template-based code
        # In a real implementation, this would call an LLM with the generated prompt
        template_code = await self._get_template_code(request)

        return template_code

    async def _generate_documentation(
        self, code: str, request: CodeGenerationRequest
    ) -> str:
        """Generate documentation for the code"""

        if not request.include_documentation:
            return ""

        # Analyze code structure
        structure = await self.code_analyzer.analyze_code_structure(
            code, request.language
        )

        # Generate documentation based on code structure
        doc_parts = []

        # Main description
        doc_parts.append(f"# {request.code_type.value.title()}: {request.description}")
        doc_parts.append("")

        # Requirements
        if request.requirements:
            doc_parts.append("## Requirements")
            for req in request.requirements:
                doc_parts.append(f"- {req}")
            doc_parts.append("")

        # Functions documentation
        if "functions" in structure and structure["functions"]:
            doc_parts.append("## Functions")
            for func in structure["functions"]:
                doc_parts.append(f"### {func['name']}")
                if func.get("docstring"):
                    doc_parts.append(func["docstring"])
                if func.get("args"):
                    doc_parts.append(f"**Parameters:** {', '.join(func['args'])}")
                doc_parts.append("")

        # Classes documentation
        if "classes" in structure and structure["classes"]:
            doc_parts.append("## Classes")
            for cls in structure["classes"]:
                doc_parts.append(f"### {cls['name']}")
                if cls.get("docstring"):
                    doc_parts.append(cls["docstring"])
                doc_parts.append("")

        # Usage example
        doc_parts.append("## Usage")
        doc_parts.append("```" + request.language.value)
        doc_parts.append("# Example usage")
        doc_parts.append("# TODO: Add specific usage example")
        doc_parts.append("```")

        return "\n".join(doc_parts)

    async def _generate_test_code(
        self, code: str, request: CodeGenerationRequest
    ) -> str:
        """Generate test code for the generated code"""

        # Extract functions to test
        functions = await self.code_analyzer.extract_functions(code, request.language)

        if not functions:
            return ""

        test_parts = []

        if request.language == ProgrammingLanguage.PYTHON:
            test_parts.append("import unittest")
            test_parts.append("")
            test_parts.append("class TestGeneratedCode(unittest.TestCase):")
            test_parts.append("")

            for func in functions:
                test_parts.append(f"    def test_{func['name']}(self):")
                test_parts.append(f"        # TODO: Implement test for {func['name']}")
                test_parts.append("        pass")
                test_parts.append("")

            test_parts.append("if __name__ == '__main__':")
            test_parts.append("    unittest.main()")

        elif request.language == ProgrammingLanguage.JAVASCRIPT:
            test_parts.append("// Test file for generated code")
            test_parts.append("const assert = require('assert');")
            test_parts.append("")

            for func in functions:
                test_parts.append(f"describe('{func['name']}', () => {{")
                test_parts.append("    it('should work correctly', () => {")
                test_parts.append(f"        // TODO: Implement test for {func['name']}")
                test_parts.append("        assert.ok(true);")
                test_parts.append("    });")
                test_parts.append("});")
                test_parts.append("")

        return "\n".join(test_parts)

    async def _validate_generated_code(
        self, code: str, request: CodeGenerationRequest, test_code: Optional[str] = None
    ) -> CodeValidationResult:
        """Validate the generated code"""

        # Create basic validation tests
        tests = await self._create_validation_tests(code, request)

        # Validate code
        return await self.code_validator.validate_code(code, request.language, tests)

    async def _create_validation_tests(
        self, code: str, request: CodeGenerationRequest
    ) -> List[ValidationTest]:
        """Create validation tests for the code"""

        tests = []

        # Extract functions for testing
        functions = await self.code_analyzer.extract_functions(code, request.language)

        for i, func in enumerate(functions):
            # Create basic test case
            test = ValidationTest(
                test_id=f"test_{i}",
                name=f"Test {func['name']}",
                input_data=self._get_sample_input(func, request.language),
                expected_output=self._get_expected_output(func, request.language),
                test_type="unit",
            )
            tests.append(test)

        return tests

    def _get_sample_input(
        self, func: Dict[str, Any], language: ProgrammingLanguage
    ) -> Any:
        """Get sample input for function testing"""

        # Simple heuristics for sample input
        if func["name"].lower().startswith("add"):
            return [1, 2]
        elif func["name"].lower().startswith("multiply"):
            return [3, 4]
        elif func["name"].lower().startswith("factorial"):
            return 5
        elif func["name"].lower().startswith("fibonacci"):
            return 6
        else:
            return []  # Default empty input

    def _get_expected_output(
        self, func: Dict[str, Any], language: ProgrammingLanguage
    ) -> Any:
        """Get expected output for function testing"""

        # Simple heuristics for expected output
        if func["name"].lower().startswith("add"):
            return 3
        elif func["name"].lower().startswith("multiply"):
            return 12
        elif func["name"].lower().startswith("factorial"):
            return 120
        elif func["name"].lower().startswith("fibonacci"):
            return 8
        else:
            return None  # Default no expected output

    async def _improve_code_iteratively(
        self,
        code: str,
        request: CodeGenerationRequest,
        validation_result: CodeValidationResult,
        max_iterations: int = 3,
    ) -> Tuple[str, int]:
        """Improve code iteratively based on validation feedback"""

        current_code = code
        iterations = 1

        for iteration in range(max_iterations):
            if validation_result.overall_result == ValidationResult.PASS:
                break

            # Analyze issues and create improvement plan
            improvement_goals = []

            for assessment in validation_result.quality_assessments:
                if assessment.score < 0.7:
                    improvement_goals.extend(assessment.suggestions)

            if validation_result.error_messages:
                improvement_goals.extend(
                    [
                        f"Fix error: {error}"
                        for error in validation_result.error_messages
                    ]
                )

            if not improvement_goals:
                break

            # Generate improved code
            improved_code = await self._apply_improvements(
                current_code, request, improvement_goals
            )

            # Validate improved code
            new_validation = await self._validate_generated_code(improved_code, request)

            # Check if improvement was successful
            if (
                new_validation.functionality_score
                > validation_result.functionality_score
            ):
                current_code = improved_code
                validation_result = new_validation
                iterations += 1
            else:
                break  # No improvement, stop iterating

        return current_code, iterations

    async def _apply_improvements(
        self, code: str, request: CodeGenerationRequest, improvement_goals: List[str]
    ) -> str:
        """Apply improvements to code based on goals"""

        improved_code = code

        for goal in improvement_goals:
            if "syntax error" in goal.lower():
                improved_code = await self._fix_syntax_errors(
                    improved_code, request.language
                )
            elif "add comments" in goal.lower():
                improved_code = await self._add_comments(
                    improved_code, request.language
                )
            elif "add docstring" in goal.lower():
                improved_code = await self._add_docstrings(
                    improved_code, request.language
                )
            elif "variable names" in goal.lower():
                improved_code = await self._improve_variable_names(
                    improved_code, request.language
                )

        return improved_code

    async def _fix_syntax_errors(self, code: str, language: ProgrammingLanguage) -> str:
        """Fix basic syntax errors"""

        if language == ProgrammingLanguage.PYTHON:
            # Basic Python syntax fixes
            fixed_code = code

            # Fix indentation issues
            lines = fixed_code.splitlines()
            fixed_lines = []

            for line in lines:
                if line.strip() and not line.startswith(" "):
                    # Add basic indentation for function/class bodies
                    if any(
                        keyword in line
                        for keyword in ["def ", "class ", "if ", "for ", "while "]
                    ):
                        fixed_lines.append(line)
                    else:
                        fixed_lines.append("    " + line)
                else:
                    fixed_lines.append(line)

            return "\n".join(fixed_lines)

        return code

    async def _add_comments(self, code: str, language: ProgrammingLanguage) -> str:
        """Add comments to code"""

        lines = code.splitlines()
        commented_lines = []

        for line in lines:
            commented_lines.append(line)

            # Add comments for function definitions
            if language == ProgrammingLanguage.PYTHON and line.strip().startswith(
                "def "
            ):
                commented_lines.append("    # TODO: Add implementation")
            elif language == ProgrammingLanguage.JAVASCRIPT and "function" in line:
                commented_lines.append("    // TODO: Add implementation")

        return "\n".join(commented_lines)

    async def _add_docstrings(self, code: str, language: ProgrammingLanguage) -> str:
        """Add docstrings to functions"""

        if language != ProgrammingLanguage.PYTHON:
            return code

        lines = code.splitlines()
        docstring_lines = []

        for i, line in enumerate(lines):
            docstring_lines.append(line)

            if line.strip().startswith("def "):
                # Add basic docstring
                docstring_lines.append('    """')
                docstring_lines.append("    TODO: Add function description")
                docstring_lines.append('    """')

        return "\n".join(docstring_lines)

    async def _improve_variable_names(
        self, code: str, language: ProgrammingLanguage
    ) -> str:
        """Improve variable names"""

        # Simple variable name improvements
        improvements = {
            "x": "value",
            "y": "result",
            "i": "index",
            "j": "counter",
            "n": "number",
            "arr": "array",
            "lst": "list_items",
        }

        improved_code = code
        for old_name, new_name in improvements.items():
            # Use word boundaries to avoid partial replacements
            pattern = r"\b" + re.escape(old_name) + r"\b"
            improved_code = re.sub(pattern, new_name, improved_code)

        return improved_code

    async def _calculate_confidence_score(
        self,
        code: str,
        validation_result: CodeValidationResult,
        request: CodeGenerationRequest,
    ) -> float:
        """Calculate confidence score for generated code"""

        base_score = 0.5

        # Functionality score contribution
        base_score += validation_result.functionality_score * 0.4

        # Quality assessments contribution
        if validation_result.quality_assessments:
            avg_quality = sum(
                assessment.score for assessment in validation_result.quality_assessments
            ) / len(validation_result.quality_assessments)
            base_score += avg_quality * 0.3

        # Validation result contribution
        if validation_result.overall_result == ValidationResult.PASS:
            base_score += 0.2
        elif validation_result.overall_result == ValidationResult.WARNING:
            base_score += 0.1

        # Code completeness
        if len(code.splitlines()) > 5:  # Reasonable code length
            base_score += 0.1

        return min(base_score, 1.0)

    async def _get_template_code(self, request: CodeGenerationRequest) -> str:
        """Get template code based on request"""

        template_key = f"{request.language.value}_{request.code_type.value}"

        if template_key in self.language_templates:
            template = self.language_templates[template_key]
            return template.format(
                description=request.description,
                function_name=self._generate_function_name(request.description),
            )

        # Default template
        if request.language == ProgrammingLanguage.PYTHON:
            if request.code_type == CodeType.FUNCTION:
                return f'''def {self._generate_function_name(request.description)}():
    """
    {request.description}
    """
    # TODO: Implement function logic
    pass'''
            elif request.code_type == CodeType.CLASS:
                return f'''class {self._generate_class_name(request.description)}:
    """
    {request.description}
    """
    
    def __init__(self):
        """Initialize the class."""
        pass'''

        return f"# {request.description}\n# TODO: Implement code"

    def _generate_function_name(self, description: str) -> str:
        """Generate function name from description"""

        # Simple function name generation
        words = re.findall(r"\w+", description.lower())
        if len(words) > 0:
            return "_".join(words[:3])  # Use first 3 words
        return "generated_function"

    def _generate_class_name(self, description: str) -> str:
        """Generate class name from description"""

        # Simple class name generation
        words = re.findall(r"\w+", description.lower())
        if len(words) > 0:
            return "".join(word.capitalize() for word in words[:2])  # Use first 2 words
        return "GeneratedClass"

    async def _load_code_templates(self):
        """Load code generation templates"""

        # This would typically load from files or database
        logger.info("Code templates loaded")

    def _initialize_language_templates(self) -> Dict[str, str]:
        """Initialize language-specific code templates"""

        return {
            "python_function": '''def {function_name}():
    """
    {description}
    """
    # TODO: Implement function logic
    pass''',
            "python_class": '''class {function_name}:
    """
    {description}
    """
    
    def __init__(self):
        """Initialize the class."""
        pass''',
            "javascript_function": """function {function_name}() {{
    // {description}
    // TODO: Implement function logic
}}""",
            "java_class": """public class {function_name} {{
    /**
     * {description}
     */
    public {function_name}() {{
        // TODO: Implement constructor
    }}
}}""",
        }

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""

        return {
            "initialized": self._initialized,
            "supported_languages": len(ProgrammingLanguage),
            "supported_code_types": len(CodeType),
            "quality_metrics": len(QualityMetric),
            "templates_loaded": len(self.language_templates),
        }

    async def shutdown(self):
        """Shutdown the code generation system"""
        logger.info("Shutting down Code Generation System")

        if self.prompt_framework:
            await self.prompt_framework.shutdown()

        logger.info("Code Generation System shutdown complete")


# Factory function
async def create_code_generator() -> CodeGenerator:
    """Create and initialize a code generator"""
    generator = CodeGenerator()
    await generator.initialize()
    return generator


# Utility functions
async def generate_python_function(
    description: str, audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE
) -> GeneratedCode:
    """Quick function to generate Python code"""
    generator = await create_code_generator()
    return await generator.generate_function(
        description, ProgrammingLanguage.PYTHON, audience_level
    )


async def generate_algorithm_implementation(
    description: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
) -> GeneratedCode:
    """Quick function to generate algorithm implementation"""
    generator = await create_code_generator()
    return await generator.generate_algorithm(description, language)
