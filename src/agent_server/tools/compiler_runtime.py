"""
Compiler/Runtime Tool
Sandboxed code execution environment with Pass@k evaluation and security scanning
"""

from typing import Dict, Any, List, Optional
import asyncio
import time
import tempfile
import os
import shutil
import re
import ast
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import signal
from contextlib import contextmanager

from .registry import (
    BaseTool,
    ToolResult,
    ExecutionContext,
    ToolCapabilities,
    ResourceRequirements,
    ToolCapability,
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LanguageConfig:
    """Configuration for a programming language"""

    name: str
    file_extension: str
    compile_command: Optional[str] = None
    run_command: str = ""
    interpreter: Optional[str] = None
    timeout_seconds: int = 30
    memory_limit_mb: int = 256
    allowed_imports: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    security_level: str = "medium"  # low, medium, high


@dataclass
class CodeExecutionResult:
    """Result of code execution"""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    memory_used_mb: float
    compilation_output: Optional[str] = None
    security_violations: List[str] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    """Test case for code evaluation"""

    input_data: Any
    expected_output: Any
    description: str = ""
    timeout_seconds: int = 10
    test_type: str = "functional"  # functional, performance, edge_case


@dataclass
class PassAtKResult:
    """Result of Pass@k evaluation"""

    k_values: List[int]
    pass_rates: Dict[int, float]
    total_test_cases: int
    successful_executions: int
    failed_executions: int
    compilation_failures: int
    runtime_errors: int
    timeout_errors: int
    detailed_results: List[Dict[str, Any]]


class SecurityScanner:
    """Scans code for security vulnerabilities and dangerous patterns"""

    def __init__(self):
        # Dangerous patterns by language
        self.dangerous_patterns = {
            "python": [
                r"import\s+os",
                r"import\s+subprocess",
                r"import\s+sys",
                r"__import__",
                r"eval\s*\(",
                r"exec\s*\(",
                r"compile\s*\(",
                r"open\s*\(",
                r"file\s*\(",
                r"input\s*\(",
                r"raw_input\s*\(",
                r"\.system\s*\(",
                r"\.popen\s*\(",
                r"\.spawn\s*\(",
            ],
            "javascript": [
                r"require\s*\(",
                r"import\s+.*from",
                r"eval\s*\(",
                r"Function\s*\(",
                r"setTimeout\s*\(",
                r"setInterval\s*\(",
                r"process\.",
                r"global\.",
                r"window\.",
                r"document\.",
            ],
            "java": [
                r"import\s+java\.io",
                r"import\s+java\.nio",
                r"import\s+java\.net",
                r"import\s+java\.lang\.reflect",
                r"Runtime\.getRuntime",
                r"ProcessBuilder",
                r"System\.exit",
                r"System\.getProperty",
                r"Class\.forName",
            ],
            "cpp": [
                r"#include\s*<fstream>",
                r"#include\s*<cstdlib>",
                r"system\s*\(",
                r"popen\s*\(",
                r"exec\w*\s*\(",
                r"fork\s*\(",
                r"malloc\s*\(",
                r"free\s*\(",
            ],
        }

        # Allowed safe imports/modules
        self.safe_imports = {
            "python": [
                "math",
                "random",
                "datetime",
                "json",
                "re",
                "collections",
                "itertools",
                "functools",
                "operator",
                "string",
                "decimal",
                "fractions",
                "statistics",
                "typing",
                "dataclasses",
            ],
            "javascript": [
                "Math",
                "Date",
                "JSON",
                "Array",
                "Object",
                "String",
                "Number",
            ],
        }

    def scan_code(self, code: str, language: str) -> Dict[str, Any]:
        """Scan code for security issues"""

        violations = []
        warnings = []
        risk_level = "low"

        language_lower = language.lower()

        # Check for dangerous patterns
        if language_lower in self.dangerous_patterns:
            for pattern in self.dangerous_patterns[language_lower]:
                matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
                if matches:
                    violations.append(f"Dangerous pattern detected: {pattern}")
                    risk_level = "high"

        # Language-specific analysis
        if language_lower == "python":
            violations.extend(self._scan_python_specific(code))
        elif language_lower == "javascript":
            violations.extend(self._scan_javascript_specific(code))

        # Check for suspicious keywords
        suspicious_keywords = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "admin",
            "root",
            "sudo",
            "chmod",
            "rm -rf",
        ]

        for keyword in suspicious_keywords:
            if keyword.lower() in code.lower():
                warnings.append(f"Suspicious keyword found: {keyword}")

        # Determine overall risk level
        if violations:
            risk_level = "high"
        elif warnings:
            risk_level = "medium"

        return {
            "violations": violations,
            "warnings": warnings,
            "risk_level": risk_level,
            "safe_to_execute": len(violations) == 0,
            "scan_timestamp": datetime.now().isoformat(),
        }

    def _scan_python_specific(self, code: str) -> List[str]:
        """Python-specific security scanning"""
        violations = []

        try:
            # Parse AST for deeper analysis
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["eval", "exec", "compile", "__import__"]:
                            violations.append(
                                f"Dangerous function call: {node.func.id}"
                            )

                # Check for dangerous imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["os", "subprocess", "sys"]:
                            violations.append(f"Dangerous import: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module in ["os", "subprocess", "sys"]:
                        violations.append(f"Dangerous import from: {node.module}")

        except SyntaxError:
            # If code has syntax errors, we'll catch it during compilation
            pass
        except Exception as e:
            logger.warning(f"AST parsing failed: {e}")

        return violations

    def _scan_javascript_specific(self, code: str) -> List[str]:
        """JavaScript-specific security scanning"""
        violations = []

        # Check for Node.js specific patterns
        if "require(" in code:
            violations.append("Node.js require() detected")

        # Check for browser-specific dangerous patterns
        if any(pattern in code for pattern in ["document.", "window.", "localStorage"]):
            violations.append("Browser DOM access detected")

        return violations


class SandboxManager:
    """Manages sandboxed execution environments"""

    def __init__(self, base_temp_dir: Optional[str] = None):
        default_temp_dir = "data/temp"
        self.base_temp_dir = base_temp_dir or default_temp_dir
        # Ensure temp directory exists
        os.makedirs(self.base_temp_dir, exist_ok=True)
        self.active_sandboxes: Dict[str, str] = {}

    @contextmanager
    def create_sandbox(self, sandbox_id: str):
        """Create a temporary sandbox directory"""

        sandbox_path = os.path.join(self.base_temp_dir, f"sandbox_{sandbox_id}")

        try:
            # Create sandbox directory
            os.makedirs(sandbox_path, exist_ok=True)
            self.active_sandboxes[sandbox_id] = sandbox_path

            logger.debug(f"Created sandbox: {sandbox_path}")
            yield sandbox_path

        finally:
            # Cleanup sandbox
            try:
                if os.path.exists(sandbox_path):
                    shutil.rmtree(sandbox_path)
                    logger.debug(f"Cleaned up sandbox: {sandbox_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox {sandbox_path}: {e}")

            # Remove from active sandboxes
            if sandbox_id in self.active_sandboxes:
                del self.active_sandboxes[sandbox_id]

    def cleanup_all_sandboxes(self):
        """Cleanup all active sandboxes"""

        for sandbox_id, sandbox_path in list(self.active_sandboxes.items()):
            try:
                if os.path.exists(sandbox_path):
                    shutil.rmtree(sandbox_path)
                    logger.debug(f"Cleaned up sandbox: {sandbox_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox {sandbox_path}: {e}")

        self.active_sandboxes.clear()


class LanguageExecutor:
    """Executes code in different programming languages"""

    def __init__(self):
        self.language_configs = self._initialize_language_configs()
        self.security_scanner = SecurityScanner()
        self.sandbox_manager = SandboxManager()

    def _initialize_language_configs(self) -> Dict[str, LanguageConfig]:
        """Initialize supported language configurations"""

        configs = {}

        # Python
        configs["python"] = LanguageConfig(
            name="python",
            file_extension=".py",
            run_command="python {filename}",
            interpreter="python",
            timeout_seconds=30,
            memory_limit_mb=256,
            allowed_imports=["math", "random", "datetime", "json", "re"],
            forbidden_patterns=["import os", "import subprocess", "eval(", "exec("],
            security_level="medium",
        )

        # JavaScript (Node.js)
        configs["javascript"] = LanguageConfig(
            name="javascript",
            file_extension=".js",
            run_command="node {filename}",
            interpreter="node",
            timeout_seconds=30,
            memory_limit_mb=256,
            forbidden_patterns=["require(", "process.", "global."],
            security_level="medium",
        )

        # Java
        configs["java"] = LanguageConfig(
            name="java",
            file_extension=".java",
            compile_command="javac {filename}",
            run_command="java {classname}",
            timeout_seconds=45,  # Longer for compilation
            memory_limit_mb=512,
            forbidden_patterns=["import java.io", "Runtime.getRuntime", "System.exit"],
            security_level="high",
        )

        # C++
        configs["cpp"] = LanguageConfig(
            name="cpp",
            file_extension=".cpp",
            compile_command="g++ -o {output} {filename} -std=c++17 -Wall",
            run_command="./{output}",
            timeout_seconds=45,
            memory_limit_mb=512,
            forbidden_patterns=["#include <fstream>", "system(", "popen("],
            security_level="high",
        )

        return configs

    async def execute_code(
        self,
        code: str,
        language: str,
        test_cases: List[TestCase] = None,
        sandbox_id: Optional[str] = None,
    ) -> CodeExecutionResult:
        """Execute code in a sandboxed environment"""

        if language not in self.language_configs:
            raise ValueError(f"Unsupported language: {language}")

        config = self.language_configs[language]

        # Security scan
        security_result = self.security_scanner.scan_code(code, language)
        if not security_result["safe_to_execute"]:
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr="Code execution blocked due to security violations",
                return_code=-1,
                execution_time=0.0,
                memory_used_mb=0.0,
                security_violations=security_result["violations"],
            )

        # Generate sandbox ID if not provided
        if not sandbox_id:
            sandbox_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:8]

        # Execute in sandbox
        with self.sandbox_manager.create_sandbox(sandbox_id) as sandbox_path:
            return await self._execute_in_sandbox(
                code, config, sandbox_path, test_cases
            )

    async def _execute_in_sandbox(
        self,
        code: str,
        config: LanguageConfig,
        sandbox_path: str,
        test_cases: List[TestCase] = None,
    ) -> CodeExecutionResult:
        """Execute code within a sandbox"""

        start_time = time.time()

        try:
            # Create source file
            source_filename = f"main{config.file_extension}"
            source_path = os.path.join(sandbox_path, source_filename)

            with open(source_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Compilation step (if needed)
            compilation_output = None
            if config.compile_command:
                compilation_result = await self._run_command(
                    config.compile_command.format(
                        filename=source_filename,
                        output="main",
                        classname="Main",  # For Java
                    ),
                    sandbox_path,
                    config.timeout_seconds,
                )

                compilation_output = (
                    compilation_result["stdout"] + compilation_result["stderr"]
                )

                if compilation_result["return_code"] != 0:
                    return CodeExecutionResult(
                        success=False,
                        stdout=compilation_result["stdout"],
                        stderr=compilation_result["stderr"],
                        return_code=compilation_result["return_code"],
                        execution_time=time.time() - start_time,
                        memory_used_mb=0.0,
                        compilation_output=compilation_output,
                    )

            # Execution step
            if test_cases:
                # Run with test cases
                return await self._execute_with_test_cases(
                    config, sandbox_path, test_cases, start_time, compilation_output
                )
            else:
                # Simple execution
                run_command = config.run_command.format(
                    filename=source_filename, output="main", classname="Main"
                )

                execution_result = await self._run_command(
                    run_command, sandbox_path, config.timeout_seconds
                )

                return CodeExecutionResult(
                    success=execution_result["return_code"] == 0,
                    stdout=execution_result["stdout"],
                    stderr=execution_result["stderr"],
                    return_code=execution_result["return_code"],
                    execution_time=time.time() - start_time,
                    memory_used_mb=execution_result.get("memory_mb", 0.0),
                    compilation_output=compilation_output,
                )

        except Exception as e:
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time,
                memory_used_mb=0.0,
            )

    async def _execute_with_test_cases(
        self,
        config: LanguageConfig,
        sandbox_path: str,
        test_cases: List[TestCase],
        start_time: float,
        compilation_output: Optional[str],
    ) -> CodeExecutionResult:
        """Execute code with multiple test cases"""

        all_stdout = []
        all_stderr = []
        all_success = True
        total_memory = 0.0

        for i, test_case in enumerate(test_cases):
            # Create input file if needed
            if test_case.input_data:
                input_file = os.path.join(sandbox_path, f"input_{i}.txt")
                with open(input_file, "w") as f:
                    if isinstance(test_case.input_data, str):
                        f.write(test_case.input_data)
                    else:
                        f.write(str(test_case.input_data))

            # Run command
            run_command = config.run_command.format(
                filename=f"main{config.file_extension}", output="main", classname="Main"
            )

            # Add input redirection if needed
            if test_case.input_data:
                run_command += f" < input_{i}.txt"

            execution_result = await self._run_command(
                run_command, sandbox_path, test_case.timeout_seconds
            )

            all_stdout.append(f"Test {i+1}: {execution_result['stdout']}")
            all_stderr.append(f"Test {i+1}: {execution_result['stderr']}")

            if execution_result["return_code"] != 0:
                all_success = False

            total_memory += execution_result.get("memory_mb", 0.0)

        return CodeExecutionResult(
            success=all_success,
            stdout="\n".join(all_stdout),
            stderr="\n".join(all_stderr),
            return_code=0 if all_success else 1,
            execution_time=time.time() - start_time,
            memory_used_mb=total_memory / len(test_cases) if test_cases else 0.0,
            compilation_output=compilation_output,
        )

    async def _run_command(
        self, command: str, working_dir: str, timeout_seconds: int
    ) -> Dict[str, Any]:
        """Run a command with timeout and resource monitoring"""

        try:
            # Create process
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_seconds
                )

                return {
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "return_code": process.returncode,
                    "memory_mb": 0.0,  # Simplified - would need psutil for real monitoring
                }

            except asyncio.TimeoutError:
                # Kill process group
                if os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()

                return {
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout_seconds} seconds",
                    "return_code": -1,
                    "memory_mb": 0.0,
                }

        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Command execution failed: {str(e)}",
                "return_code": -1,
                "memory_mb": 0.0,
            }


class PassAtKEvaluator:
    """Evaluates code using Pass@k metrics"""

    def __init__(self, language_executor: LanguageExecutor):
        self.language_executor = language_executor

    async def evaluate_pass_at_k(
        self,
        code_samples: List[str],
        test_cases: List[TestCase],
        language: str,
        k_values: List[int] = [1, 3, 5],
    ) -> PassAtKResult:
        """Evaluate Pass@k metrics for code samples"""

        if not code_samples:
            raise ValueError("No code samples provided")

        if not test_cases:
            raise ValueError("No test cases provided")

        logger.info(
            f"Evaluating Pass@k for {len(code_samples)} code samples with {len(test_cases)} test cases"
        )

        # Execute all code samples
        execution_results = []
        successful_executions = 0
        failed_executions = 0
        compilation_failures = 0
        runtime_errors = 0
        timeout_errors = 0

        for i, code in enumerate(code_samples):
            logger.debug(f"Executing code sample {i+1}/{len(code_samples)}")

            result = await self.language_executor.execute_code(
                code=code,
                language=language,
                test_cases=test_cases,
                sandbox_id=f"passk_{i}",
            )

            # Categorize results
            if result.success:
                successful_executions += 1
            else:
                failed_executions += 1

                if result.compilation_output and result.return_code != 0:
                    compilation_failures += 1
                elif "timed out" in result.stderr.lower():
                    timeout_errors += 1
                else:
                    runtime_errors += 1

            # Store detailed result
            detailed_result = {
                "sample_index": i,
                "success": result.success,
                "execution_time": result.execution_time,
                "memory_used_mb": result.memory_used_mb,
                "return_code": result.return_code,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
                "has_compilation_output": result.compilation_output is not None,
                "security_violations": len(result.security_violations),
                "test_results": self._analyze_test_results(result, test_cases),
            }

            execution_results.append(detailed_result)

        # Calculate Pass@k rates
        pass_rates = {}
        for k in k_values:
            if k <= len(code_samples):
                pass_rate = self._calculate_pass_at_k_rate(execution_results, k)
                pass_rates[k] = pass_rate
            else:
                pass_rates[k] = 0.0  # Can't calculate Pass@k for k > number of samples

        return PassAtKResult(
            k_values=k_values,
            pass_rates=pass_rates,
            total_test_cases=len(test_cases),
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            compilation_failures=compilation_failures,
            runtime_errors=runtime_errors,
            timeout_errors=timeout_errors,
            detailed_results=execution_results,
        )

    def _analyze_test_results(
        self, execution_result: CodeExecutionResult, test_cases: List[TestCase]
    ) -> Dict[str, Any]:
        """Analyze test case results from execution"""

        # This is a simplified analysis - in practice, you'd need more sophisticated
        # output parsing to determine which test cases passed/failed

        return {
            "total_tests": len(test_cases),
            "estimated_passed": len(test_cases) if execution_result.success else 0,
            "estimated_failed": 0 if execution_result.success else len(test_cases),
            "analysis_method": "simplified",  # Indicates this is basic analysis
        }

    def _calculate_pass_at_k_rate(
        self, execution_results: List[Dict[str, Any]], k: int
    ) -> float:
        """Calculate Pass@k rate"""

        # Pass@k = Probability that at least one of the top k samples passes all tests
        # For simplicity, we'll use the success rate of the top k samples

        if not execution_results or k <= 0:
            return 0.0

        # Sort by success (successful first) and then by execution time
        sorted_results = sorted(
            execution_results, key=lambda x: (-int(x["success"]), x["execution_time"])
        )

        # Take top k results
        top_k_results = sorted_results[:k]

        # Check if any of the top k passed
        any_passed = any(result["success"] for result in top_k_results)

        return 1.0 if any_passed else 0.0


class CompilerRuntimeTool(BaseTool):
    """Compiler/Runtime tool for code validation and Pass@k evaluation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.language_executor = LanguageExecutor()
        self.pass_at_k_evaluator = PassAtKEvaluator(self.language_executor)

        # Configuration
        self.max_code_length = config.get("max_code_length", 10000) if config else 10000
        self.max_execution_time = config.get("max_execution_time", 60) if config else 60
        self.enable_security_scanning = (
            config.get("enable_security_scanning", True) if config else True
        )

    async def initialize(self):
        """Initialize the compiler/runtime tool"""
        await super().initialize()

        # Check for required interpreters/compilers
        available_languages = await self._check_available_languages()

        logger.info(
            "Compiler/Runtime Tool initialized",
            available_languages=available_languages,
            security_scanning_enabled=self.enable_security_scanning,
        )

    async def _check_available_languages(self) -> List[str]:
        """Check which language interpreters/compilers are available"""

        available = []

        # Check Python
        try:
            result = await asyncio.create_subprocess_exec(
                "python",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            if result.returncode == 0:
                available.append("python")
        except:
            pass

        # Check Node.js
        try:
            result = await asyncio.create_subprocess_exec(
                "node",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            if result.returncode == 0:
                available.append("javascript")
        except:
            pass

        # Check Java
        try:
            result = await asyncio.create_subprocess_exec(
                "javac",
                "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            if result.returncode == 0:
                available.append("java")
        except:
            pass

        # Check GCC (C++)
        try:
            result = await asyncio.create_subprocess_exec(
                "g++",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            if result.returncode == 0:
                available.append("cpp")
        except:
            pass

        return available

    async def execute(
        self, parameters: Dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Execute code compilation and runtime evaluation"""

        start_time = time.time()

        try:
            # Extract parameters
            code = parameters.get("code", "")
            language = parameters.get("language", "python").lower()
            evaluation_type = parameters.get(
                "evaluation_type", "simple"
            )  # simple, pass_at_k
            test_cases_data = parameters.get("test_cases", [])
            k_values = parameters.get("k_values", [1, 3, 5])
            code_samples = parameters.get(
                "code_samples", [code]
            )  # For Pass@k evaluation

            logger.info(
                "Executing code compilation and runtime evaluation",
                language=language,
                evaluation_type=evaluation_type,
                code_length=len(code),
                test_cases_count=len(test_cases_data),
                session_id=context.session_id,
            )

            # Validate inputs
            if not code and not code_samples:
                raise ValueError("No code provided for evaluation")

            if len(code) > self.max_code_length:
                raise ValueError(
                    f"Code length exceeds maximum of {self.max_code_length} characters"
                )

            if language not in self.language_executor.language_configs:
                raise ValueError(f"Unsupported language: {language}")

            # Parse test cases
            test_cases = []
            for tc_data in test_cases_data:
                test_case = TestCase(
                    input_data=tc_data.get("input"),
                    expected_output=tc_data.get("expected_output"),
                    description=tc_data.get("description", ""),
                    timeout_seconds=tc_data.get("timeout", 10),
                    test_type=tc_data.get("type", "functional"),
                )
                test_cases.append(test_case)

            # Execute based on evaluation type
            if evaluation_type == "pass_at_k":
                result_data = await self._execute_pass_at_k_evaluation(
                    code_samples, test_cases, language, k_values, context
                )
            else:
                result_data = await self._execute_simple_evaluation(
                    code, test_cases, language, context
                )

            execution_time = time.time() - start_time
            result_data["processing_time"] = execution_time

            # Calculate quality and confidence scores
            quality_score = self._calculate_quality_score(result_data, evaluation_type)
            confidence_score = self._calculate_confidence_score(result_data, language)

            result = ToolResult(
                data=result_data,
                metadata={
                    "tool": "compiler_runtime",
                    "version": "1.0.0",
                    "language": language,
                    "evaluation_type": evaluation_type,
                    "session_id": context.session_id,
                    "security_scanning_enabled": self.enable_security_scanning,
                },
                execution_time=execution_time,
                success=result_data.get("overall_success", True),
                resource_usage={
                    "cpu_usage": 0.8,  # Code execution is CPU intensive
                    "memory_usage_mb": result_data.get("peak_memory_mb", 128),
                    "disk_usage_mb": 10,  # For temporary files
                },
                quality_score=quality_score,
                confidence_score=confidence_score,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Code compilation/runtime evaluation failed",
                error=str(e),
                session_id=context.session_id,
            )

            return ToolResult(
                data=None,
                metadata={
                    "tool": "compiler_runtime",
                    "session_id": context.session_id,
                    "error_type": type(e).__name__,
                },
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                quality_score=0.0,
                confidence_score=0.0,
            )

    async def _execute_simple_evaluation(
        self,
        code: str,
        test_cases: List[TestCase],
        language: str,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute simple code evaluation"""

        # Execute code
        execution_result = await self.language_executor.execute_code(
            code=code, language=language, test_cases=test_cases
        )

        # Analyze results
        result_data = {
            "evaluation_type": "simple",
            "language": language,
            "execution_result": {
                "success": execution_result.success,
                "stdout": execution_result.stdout,
                "stderr": execution_result.stderr,
                "return_code": execution_result.return_code,
                "execution_time": execution_result.execution_time,
                "memory_used_mb": execution_result.memory_used_mb,
                "compilation_output": execution_result.compilation_output,
                "security_violations": execution_result.security_violations,
            },
            "code_analysis": self._analyze_code_quality(code, language),
            "test_results": {
                "total_tests": len(test_cases),
                "tests_run": len(test_cases) if execution_result.success else 0,
                "estimated_passed": len(test_cases) if execution_result.success else 0,
                "estimated_failed": 0 if execution_result.success else len(test_cases),
            },
            "overall_success": execution_result.success,
            "peak_memory_mb": execution_result.memory_used_mb,
        }

        return result_data

    async def _execute_pass_at_k_evaluation(
        self,
        code_samples: List[str],
        test_cases: List[TestCase],
        language: str,
        k_values: List[int],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute Pass@k evaluation"""

        # Run Pass@k evaluation
        pass_at_k_result = await self.pass_at_k_evaluator.evaluate_pass_at_k(
            code_samples=code_samples,
            test_cases=test_cases,
            language=language,
            k_values=k_values,
        )

        # Analyze code quality for all samples
        code_quality_analyses = []
        for i, code in enumerate(code_samples):
            analysis = self._analyze_code_quality(code, language)
            analysis["sample_index"] = i
            code_quality_analyses.append(analysis)

        # Calculate aggregate metrics
        avg_execution_time = sum(
            r["execution_time"] for r in pass_at_k_result.detailed_results
        ) / len(pass_at_k_result.detailed_results)
        avg_memory_usage = sum(
            r["memory_used_mb"] for r in pass_at_k_result.detailed_results
        ) / len(pass_at_k_result.detailed_results)

        result_data = {
            "evaluation_type": "pass_at_k",
            "language": language,
            "pass_at_k_results": {
                "k_values": pass_at_k_result.k_values,
                "pass_rates": pass_at_k_result.pass_rates,
                "total_test_cases": pass_at_k_result.total_test_cases,
                "successful_executions": pass_at_k_result.successful_executions,
                "failed_executions": pass_at_k_result.failed_executions,
                "compilation_failures": pass_at_k_result.compilation_failures,
                "runtime_errors": pass_at_k_result.runtime_errors,
                "timeout_errors": pass_at_k_result.timeout_errors,
            },
            "code_samples_analysis": {
                "total_samples": len(code_samples),
                "quality_analyses": code_quality_analyses,
                "aggregate_metrics": {
                    "avg_execution_time": avg_execution_time,
                    "avg_memory_usage_mb": avg_memory_usage,
                    "success_rate": pass_at_k_result.successful_executions
                    / len(code_samples),
                },
            },
            "detailed_results": pass_at_k_result.detailed_results,
            "overall_success": pass_at_k_result.successful_executions > 0,
            "peak_memory_mb": max(
                r["memory_used_mb"] for r in pass_at_k_result.detailed_results
            ),
        }

        return result_data

    def _analyze_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code quality metrics"""

        analysis = {
            "language": language,
            "line_count": len(code.split("\n")),
            "character_count": len(code),
            "estimated_complexity": "unknown",
        }

        # Language-specific analysis
        if language == "python":
            analysis.update(self._analyze_python_quality(code))
        elif language == "javascript":
            analysis.update(self._analyze_javascript_quality(code))
        elif language == "java":
            analysis.update(self._analyze_java_quality(code))
        elif language == "cpp":
            analysis.update(self._analyze_cpp_quality(code))

        return analysis

    def _analyze_python_quality(self, code: str) -> Dict[str, Any]:
        """Analyze Python code quality"""

        analysis = {}

        try:
            # Parse AST
            tree = ast.parse(code)

            # Count different node types
            function_count = 0
            class_count = 0
            import_count = 0
            loop_count = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_count += 1
                elif isinstance(node, (ast.For, ast.While)):
                    loop_count += 1

            analysis.update(
                {
                    "function_count": function_count,
                    "class_count": class_count,
                    "import_count": import_count,
                    "loop_count": loop_count,
                    "estimated_complexity": self._estimate_complexity(
                        function_count, class_count, loop_count
                    ),
                }
            )

        except SyntaxError as e:
            analysis["syntax_error"] = str(e)
            analysis["estimated_complexity"] = "invalid"
        except Exception as e:
            analysis["analysis_error"] = str(e)

        return analysis

    def _analyze_javascript_quality(self, code: str) -> Dict[str, Any]:
        """Analyze JavaScript code quality"""

        # Simple pattern-based analysis for JavaScript
        function_count = len(re.findall(r"function\s+\w+", code))
        arrow_function_count = len(re.findall(r"=>", code))
        class_count = len(re.findall(r"class\s+\w+", code))
        loop_count = len(re.findall(r"\b(for|while)\s*\(", code))

        return {
            "function_count": function_count,
            "arrow_function_count": arrow_function_count,
            "class_count": class_count,
            "loop_count": loop_count,
            "estimated_complexity": self._estimate_complexity(
                function_count + arrow_function_count, class_count, loop_count
            ),
        }

    def _analyze_java_quality(self, code: str) -> Dict[str, Any]:
        """Analyze Java code quality"""

        method_count = len(
            re.findall(
                r"\b(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(", code
            )
        )
        class_count = len(re.findall(r"\bclass\s+\w+", code))
        interface_count = len(re.findall(r"\binterface\s+\w+", code))
        loop_count = len(re.findall(r"\b(for|while)\s*\(", code))

        return {
            "method_count": method_count,
            "class_count": class_count,
            "interface_count": interface_count,
            "loop_count": loop_count,
            "estimated_complexity": self._estimate_complexity(
                method_count, class_count, loop_count
            ),
        }

    def _analyze_cpp_quality(self, code: str) -> Dict[str, Any]:
        """Analyze C++ code quality"""

        function_count = len(re.findall(r"\w+\s+\w+\s*\([^)]*\)\s*{", code))
        class_count = len(re.findall(r"\bclass\s+\w+", code))
        struct_count = len(re.findall(r"\bstruct\s+\w+", code))
        loop_count = len(re.findall(r"\b(for|while)\s*\(", code))
        include_count = len(re.findall(r"#include", code))

        return {
            "function_count": function_count,
            "class_count": class_count,
            "struct_count": struct_count,
            "loop_count": loop_count,
            "include_count": include_count,
            "estimated_complexity": self._estimate_complexity(
                function_count, class_count, loop_count
            ),
        }

    def _estimate_complexity(
        self, function_count: int, class_count: int, loop_count: int
    ) -> str:
        """Estimate code complexity based on counts"""

        complexity_score = function_count + class_count * 2 + loop_count

        if complexity_score <= 2:
            return "low"
        elif complexity_score <= 8:
            return "medium"
        else:
            return "high"

    def _calculate_quality_score(
        self, result_data: Dict[str, Any], evaluation_type: str
    ) -> float:
        """Calculate quality score for code evaluation"""

        base_score = 0.6

        if evaluation_type == "simple":
            # Simple evaluation scoring
            if result_data.get("overall_success", False):
                base_score += 0.3

            execution_result = result_data.get("execution_result", {})
            if not execution_result.get("security_violations", []):
                base_score += 0.1

        elif evaluation_type == "pass_at_k":
            # Pass@k evaluation scoring
            pass_rates = result_data.get("pass_at_k_results", {}).get("pass_rates", {})
            if pass_rates:
                # Use Pass@1 rate as primary quality indicator
                pass_at_1 = pass_rates.get(1, 0.0)
                base_score += pass_at_1 * 0.4

        return min(1.0, base_score)

    def _calculate_confidence_score(
        self, result_data: Dict[str, Any], language: str
    ) -> float:
        """Calculate confidence score for code evaluation"""

        base_confidence = 0.7

        # Language support factor
        if language in ["python", "javascript"]:
            base_confidence += 0.1
        elif language in ["java", "cpp"]:
            base_confidence += 0.05  # Slightly lower due to compilation complexity

        # Execution success factor
        if result_data.get("overall_success", False):
            base_confidence += 0.2

        # Security factor
        if result_data.get("evaluation_type") == "simple":
            execution_result = result_data.get("execution_result", {})
            if not execution_result.get("security_violations", []):
                base_confidence += 0.1

        return min(1.0, base_confidence)

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for compiler/runtime evaluation"""
        return {
            "name": "compiler_runtime",
            "description": "Execute and evaluate code with compilation, runtime analysis, and Pass@k metrics",
            "version": "1.0.0",
            "parameters": {
                "code": {
                    "type": "string",
                    "description": "Source code to compile and execute",
                    "required": False,
                    "maxLength": 10000,
                },
                "language": {
                    "type": "string",
                    "description": "Programming language",
                    "enum": ["python", "javascript", "java", "cpp"],
                    "default": "python",
                    "required": False,
                },
                "evaluation_type": {
                    "type": "string",
                    "description": "Type of evaluation to perform",
                    "enum": ["simple", "pass_at_k"],
                    "default": "simple",
                    "required": False,
                },
                "test_cases": {
                    "type": "array",
                    "description": "Test cases for code evaluation",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "expected_output": {"type": "string"},
                            "description": {"type": "string"},
                            "timeout": {"type": "integer", "default": 10},
                            "type": {"type": "string", "default": "functional"},
                        },
                    },
                    "required": False,
                },
                "code_samples": {
                    "type": "array",
                    "description": "Multiple code samples for Pass@k evaluation",
                    "items": {"type": "string"},
                    "required": False,
                },
                "k_values": {
                    "type": "array",
                    "description": "K values for Pass@k evaluation",
                    "items": {"type": "integer"},
                    "default": [1, 3, 5],
                    "required": False,
                },
            },
            "required_params": [],
            "returns": {
                "type": "object",
                "properties": {
                    "evaluation_type": {"type": "string"},
                    "language": {"type": "string"},
                    "execution_result": {"type": "object"},
                    "pass_at_k_results": {"type": "object"},
                    "code_analysis": {"type": "object"},
                    "code_samples_analysis": {"type": "object"},
                    "test_results": {"type": "object"},
                    "detailed_results": {"type": "array"},
                    "overall_success": {"type": "boolean"},
                    "peak_memory_mb": {"type": "number"},
                    "processing_time": {"type": "number"},
                },
            },
            "capabilities": {
                "primary": "code_execution",
                "secondary": ["code_analysis", "performance_testing"],
                "input_types": ["source_code", "test_cases"],
                "output_types": ["execution_results", "metrics", "analysis"],
                "supported_languages": ["python", "javascript", "java", "cpp"],
            },
        }

    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.CODE_GENERATION,
            secondary_capabilities=[
                ToolCapability.VALIDATION,
                ToolCapability.DATA_ANALYSIS,
            ],
            input_types=["source_code", "test_cases", "code_samples"],
            output_types=["execution_results", "pass_at_k_metrics", "code_analysis"],
            supported_formats=["python", "javascript", "java", "cpp"],
            language_support=["python", "javascript", "java", "cpp"],
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=2.0,  # Code compilation and execution can be CPU intensive
            memory_mb=1024,  # Need memory for compilation and execution
            network_bandwidth_mbps=0.0,  # No network required
            storage_mb=100,  # For temporary files and compilation artifacts
            gpu_memory_mb=0,
            max_execution_time=120,  # Allow time for compilation and multiple test runs
            concurrent_limit=2,  # Limit concurrent code executions for safety
        )

    async def cleanup(self):
        """Cleanup tool resources"""
        logger.info("Cleaning up Compiler/Runtime Tool")

        # Cleanup all sandboxes
        self.language_executor.sandbox_manager.cleanup_all_sandboxes()

        logger.info("Compiler/Runtime Tool cleanup complete")
