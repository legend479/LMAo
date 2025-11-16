#!/usr/bin/env python3
"""
Agent Server Interactive CLI
Complete testing and verification tool for all Agent Server features
"""

import asyncio
import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.markdown import Markdown
except ImportError:
    print("Installing rich package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.markdown import Markdown

console = Console()


class AgentServerInteractiveCLI:
    """Complete interactive CLI for Agent Server testing"""

    def __init__(self):
        self.agent_server = None
        self.session_id = f"cli_session_{int(time.time())}"
        self.user_id = "cli_user"
        self.session_stats = {
            "messages_processed": 0,
            "tools_executed": 0,
            "plans_created": 0,
            "errors_encountered": 0,
            "session_start": datetime.now(),
        }
        self.conversation_history = []
        self.debug_mode = True  # Enable detailed error logging

    def print_error_details(self, error: Exception, context: str = ""):
        """Print detailed error information for debugging"""
        import traceback

        console.print(
            f"\n[bold red]❌ Error in {context}[/bold red]"
            if context
            else "\n[bold red]❌ Error[/bold red]"
        )
        console.print(f"[red]Error Type: {type(error).__name__}[/red]")
        console.print(f"[red]Error Message: {str(error)}[/red]")

        if self.debug_mode:
            console.print("\n[yellow]📋 Full Traceback:[/yellow]")
            console.print("[dim]" + "=" * 70 + "[/dim]")

            # Get full traceback
            tb_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            for line in tb_lines:
                console.print(f"[dim]{line}[/dim]", end="")

            console.print("[dim]" + "=" * 70 + "[/dim]")

        self.session_stats["errors_encountered"] += 1

    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║           AGENT SERVER INTERACTIVE CLI                        ║
║     Complete Testing & Verification Tool                      ║
╠═══════════════════════════════════════════════════════════════╣
║  🤖 Orchestration | 🧠 Planning | 🔧 Tools | 💾 Memory       ║
╚═══════════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold cyan")

    def check_dependencies(self):
        """Check and install dependencies"""
        console.print("\n[yellow]🔍 Checking dependencies...[/yellow]")

        if sys.version_info < (3, 9):
            console.print("[red]❌ Python 3.9+ required[/red]")
            return False

        required_packages = ["fastapi", "redis", "langgraph", "langchain_core"]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            console.print(
                f"[yellow]Installing missing packages: {', '.join(missing_packages)}[/yellow]"
            )
            for package in missing_packages:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except:
                    console.print(f"[red]Failed to install {package}[/red]")
                    return False

        console.print("[green]✅ Dependencies OK[/green]")
        return True

    def check_redis(self):
        """Check Redis connectivity"""
        try:
            import redis

            client = redis.Redis(host="localhost", port=6379, decode_responses=True)
            client.ping()
            console.print("[green]✅ Redis running[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Redis not accessible: {str(e)[:50]}[/red]")
            return False

    async def initialize_agent_server(self):
        """Initialize the Agent Server with all enhancements"""
        if self.agent_server is not None:
            return True

        console.print("\n[yellow]🚀 Initializing Enhanced Agent Server...[/yellow]")
        console.print("[dim]Loading: Agent + RAG + Feedback + Learning systems[/dim]")

        try:
            from src.agent_server.main import AgentServer

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading components...", total=None)

                self.agent_server = AgentServer()

                progress.update(task, description="Initializing agent components...")
                await self.agent_server.initialize()

                progress.update(
                    task, description="✅ Enhanced Agent Server initialized!"
                )

            console.print("[green]✅ Enhanced Agent Server ready[/green]")
            console.print("[dim]  • Agent orchestration ✓[/dim]")
            console.print("[dim]  • RAG pipeline with enhancements ✓[/dim]")
            console.print("[dim]  • Feedback system ✓[/dim]")
            console.print("[dim]  • Learning system ✓[/dim]")
            return True

        except Exception as e:
            console.print(f"[red]❌ Agent Server initialization failed: {str(e)}[/red]")
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return False

    async def health_check(self):
        """Comprehensive system health check"""
        console.print("\n[bold blue]🏥 System Health Check[/bold blue]")

        if not self.agent_server:
            console.print("[red]❌ Agent Server not initialized[/red]")
            return False

        try:
            table = Table(title="Component Health Status", show_header=True)
            table.add_column("Component", style="cyan", width=25)
            table.add_column("Status", style="green", width=15)
            table.add_column("Details", style="dim", width=40)

            # Check Agent Server
            status = "healthy" if self.agent_server._initialized else "not_initialized"
            table.add_row(
                "Agent Server",
                f"[{'green' if status == 'healthy' else 'red'}]{status}[/]",
                "Main orchestration engine",
            )

            # Check Orchestrator
            if hasattr(self.agent_server, "orchestrator"):
                orch_status = (
                    "healthy"
                    if self.agent_server.orchestrator._initialized
                    else "not_initialized"
                )
                table.add_row(
                    "LangGraph Orchestrator",
                    f"[{'green' if orch_status == 'healthy' else 'red'}]{orch_status}[/]",
                    "Workflow management",
                )

            # Check Planning Module
            if hasattr(self.agent_server, "planning_module"):
                plan_status = (
                    "healthy"
                    if self.agent_server.planning_module._initialized
                    else "not_initialized"
                )
                table.add_row(
                    "Planning Module",
                    f"[{'green' if plan_status == 'healthy' else 'red'}]{plan_status}[/]",
                    "Task decomposition & planning",
                )

            # Check Memory Manager
            if hasattr(self.agent_server, "memory_manager"):
                mem_status = (
                    "healthy"
                    if self.agent_server.memory_manager._initialized
                    else "not_initialized"
                )
                table.add_row(
                    "Memory Manager",
                    f"[{'green' if mem_status == 'healthy' else 'red'}]{mem_status}[/]",
                    "Conversation context management",
                )

            # Check Tool Registry
            if hasattr(self.agent_server, "tool_registry"):
                tool_status = (
                    "healthy"
                    if hasattr(self.agent_server.tool_registry, "active_tools")
                    else "unknown"
                )
                tool_count = len(
                    getattr(self.agent_server.tool_registry, "active_tools", {})
                )
                table.add_row(
                    "Tool Registry",
                    f"[green]{tool_status}[/]",
                    f"{tool_count} tools registered",
                )

            console.print(table)
            return True

        except Exception as e:
            console.print(f"[red]❌ Health check failed: {str(e)}[/red]")
            return False

    async def process_message(self, message: str = None):
        """Process a message through the agent with detailed error tracking"""
        if not message:
            message = Prompt.ask("\n[cyan]Enter your message[/cyan]")

        if not message.strip():
            console.print("[red]❌ Message cannot be empty[/red]")
            return False

        console.print(f"\n[bold blue]💬 Processing: '{message}'[/bold blue]")

        # Log the start
        if self.debug_mode:
            console.print(f"[dim]Session ID: {self.session_id}[/dim]")
            console.print(f"[dim]User ID: {self.user_id}[/dim]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Step 1: Planning
                task = progress.add_task(
                    "Step 1/5: Analyzing message intent...", total=None
                )

                if self.debug_mode:
                    console.print("\n[dim]🧠 Planning Phase Starting...[/dim]")

                # Step 2: Context retrieval
                progress.update(task, description="Step 2/5: Retrieving context...")

                if self.debug_mode:
                    console.print("[dim]💾 Loading conversation context...[/dim]")

                # Step 3: Planning
                progress.update(
                    task, description="Step 3/5: Creating execution plan..."
                )

                if self.debug_mode:
                    console.print("[dim]📋 Decomposing tasks...[/dim]")

                # Step 4: Execution
                progress.update(task, description="Step 4/5: Executing plan...")

                if self.debug_mode:
                    console.print("[dim]⚙️  Execution Phase Starting...[/dim]")

                result = await self.agent_server.process_message(
                    message, self.session_id, self.user_id
                )

                # Step 5: Complete
                progress.update(task, description="Step 5/5: Finalizing response...")

                if self.debug_mode:
                    console.print("[dim]✅ Processing Complete[/dim]")

                progress.update(task, description="✅ Message processed!")

            # Check for errors in result
            if result.get("metadata", {}).get("error"):
                console.print("\n[yellow]⚠️  Processing completed with errors:[/yellow]")
                console.print(
                    f"[yellow]Error Type: {result['metadata'].get('error_type', 'Unknown')}[/yellow]"
                )

                if self.debug_mode:
                    console.print("\n[dim]Full metadata:[/dim]")
                    console.print(
                        f"[dim]{json.dumps(result.get('metadata', {}), indent=2)}[/dim]"
                    )

            # Display result
            console.print("\n[bold green]📤 Agent Response:[/bold green]")
            console.print(Panel(result["response"], border_style="green"))

            # Display metadata
            if result.get("metadata"):
                metadata_table = Table(title="Execution Metadata")
                metadata_table.add_column("Key", style="cyan")
                metadata_table.add_column("Value", style="yellow")

                for key, value in result["metadata"].items():
                    if key not in ["error", "error_type"]:
                        # Truncate long values
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        metadata_table.add_row(key, value_str)

                console.print(metadata_table)

                # Show execution path if available
                if "execution_path" in result["metadata"] and self.debug_mode:
                    console.print("\n[bold]Execution Path:[/bold]")
                    exec_path = result["metadata"]["execution_path"]
                    if isinstance(exec_path, list):
                        for step in exec_path:
                            console.print(f"  → [cyan]{step}[/cyan]")

            # Store in history
            self.conversation_history.append(
                {
                    "message": message,
                    "response": result["response"],
                    "timestamp": result["timestamp"],
                    "metadata": result["metadata"],
                }
            )

            self.session_stats["messages_processed"] += 1
            return True

        except Exception as e:
            self.print_error_details(e, "Message Processing")

            # Try to provide helpful context
            console.print("\n[yellow]💡 Debugging Tips:[/yellow]")
            console.print("  • Check if all components are initialized")
            console.print("  • Verify Redis is running")
            console.print("  • Check the execution logs above for the failure point")
            console.print("  • Try option 1 (Health Check) to verify system status")

            return False

    async def list_available_tools(self):
        """List all available tools"""
        console.print("\n[bold blue]🔧 Available Tools[/bold blue]")

        try:
            tools = await self.agent_server.get_available_tools()

            if not tools or not tools.get("tools"):
                console.print("[yellow]No tools registered[/yellow]")
                return False

            tools_list = tools.get("tools", [])

            table = Table(title=f"Registered Tools ({len(tools_list)} total)")
            table.add_column("Tool Name", style="cyan", width=25)
            table.add_column("Category", style="magenta", width=20)
            table.add_column("Description", style="dim", width=50)

            for tool in tools_list:
                table.add_row(
                    tool.get("name", "Unknown"),
                    tool.get("category", "N/A"),
                    tool.get("description", "No description")[:50] + "...",
                )

            console.print(table)
            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to list tools: {str(e)}[/red]")
            return False

    async def execute_tool(self, tool_name: str = None):
        """Execute a specific tool"""
        if not tool_name:
            tool_name = Prompt.ask("\n[cyan]Enter tool name[/cyan]")

        if not tool_name.strip():
            console.print("[red]❌ Tool name cannot be empty[/red]")
            return False

        console.print(f"\n[bold blue]⚙️  Executing Tool: {tool_name}[/bold blue]")

        # Get parameters
        console.print(
            "[yellow]Enter tool parameters (JSON format, or press Enter for empty):[/yellow]"
        )
        params_str = Prompt.ask("[cyan]Parameters[/cyan]", default="{}")

        try:
            parameters = json.loads(params_str)
        except json.JSONDecodeError:
            console.print("[red]❌ Invalid JSON format[/red]")
            return False

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Executing {tool_name}...", total=None)

                result = await self.agent_server.execute_tool(
                    tool_name, parameters, self.session_id
                )

                progress.update(task, description=f"✅ {tool_name} executed!")

            # Display result
            if result.get("status") == "success":
                console.print("\n[bold green]✅ Tool Execution Successful[/bold green]")

                result_table = Table(title="Execution Result")
                result_table.add_column("Metric", style="cyan")
                result_table.add_column("Value", style="green")

                result_table.add_row("Tool Name", result.get("tool_name", "N/A"))
                result_table.add_row("Status", result.get("status", "N/A"))
                result_table.add_row(
                    "Execution Time", f"{result.get('execution_time', 0):.3f}s"
                )

                console.print(result_table)

                # Display result data
                if result.get("result"):
                    console.print("\n[bold]Result Data:[/bold]")
                    console.print(
                        Panel(str(result["result"])[:500], border_style="blue")
                    )

            else:
                console.print(
                    f"\n[red]❌ Tool Execution Failed: {result.get('error', 'Unknown error')}[/red]"
                )

            self.session_stats["tools_executed"] += 1
            return result.get("status") == "success"

        except Exception as e:
            console.print(f"[red]❌ Tool execution failed: {str(e)}[/red]")
            return False

    async def show_execution_traces(self):
        """Show execution traces and reasoning"""
        console.print("\n[bold blue]🔍 Execution Traces & Reasoning[/bold blue]")

        if not hasattr(self.agent_server, "orchestrator"):
            console.print("[yellow]Orchestrator not available[/yellow]")
            return False

        try:
            # Get execution history
            execution_history = getattr(
                self.agent_server.orchestrator, "execution_history", {}
            )

            if not execution_history or self.session_id not in execution_history:
                console.print("[yellow]No execution history for this session[/yellow]")
                return False

            history = execution_history[self.session_id]

            for idx, execution in enumerate(history, 1):
                console.print(f"\n[bold cyan]Execution #{idx}[/bold cyan]")

                # Execution path
                if execution.execution_path:
                    console.print("\n[bold]Execution Path:[/bold]")
                    tree = Tree("🌳 Workflow")
                    for step in execution.execution_path:
                        tree.add(f"[cyan]{step}[/cyan]")
                    console.print(tree)

                # Tool results
                if execution.tool_results:
                    console.print("\n[bold]Tool Results:[/bold]")
                    for tool_result in execution.tool_results:
                        console.print(
                            f"  • [green]{tool_result.get('task_id', 'Unknown')}[/green]"
                        )

                # Metadata
                console.print("\n[bold]Metadata:[/bold]")
                metadata_table = Table()
                metadata_table.add_column("Key", style="cyan")
                metadata_table.add_column("Value", style="yellow")

                for key, value in execution.metadata.items():
                    if key not in ["execution_path", "checkpoints_count"]:
                        metadata_table.add_row(key, str(value))

                console.print(metadata_table)

                # Checkpoints
                if execution.checkpoints:
                    console.print(
                        f"\n[dim]Checkpoints: {len(execution.checkpoints)} saved[/dim]"
                    )

            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to show traces: {str(e)}[/red]")
            return False

    async def show_memory_context(self):
        """Show conversation memory and context"""
        console.print("\n[bold blue]💾 Memory & Context[/bold blue]")

        if not hasattr(self.agent_server, "memory_manager"):
            console.print("[yellow]Memory manager not available[/yellow]")
            return False

        try:
            # Get context
            context = await self.agent_server.memory_manager.get_context(
                self.session_id
            )

            # Display conversation history
            console.print("\n[bold]Conversation History:[/bold]")
            if context.message_history:
                for idx, msg in enumerate(
                    context.message_history[-5:], 1
                ):  # Last 5 messages
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")[:100]
                    timestamp = msg.get("timestamp", "N/A")

                    role_color = "cyan" if role == "user" else "green"
                    console.print(f"  [{role_color}]{role}[/]: {content}...")
            else:
                console.print("  [dim]No conversation history[/dim]")

            # Display current topic
            if context.current_topic:
                console.print(
                    f"\n[bold]Current Topic:[/bold] [yellow]{context.current_topic}[/yellow]"
                )

            # Display domain context
            if context.domain_context:
                console.print("\n[bold]Domain Context:[/bold]")
                for key, value in context.domain_context.items():
                    console.print(f"  • {key}: {value}")

            # Display user preferences
            if context.user_preferences:
                console.print("\n[bold]User Preferences:[/bold]")
                for key, value in list(context.user_preferences.items())[:5]:
                    console.print(f"  • {key}: {value}")

            # Session stats
            console.print("\n[bold]Session Statistics:[/bold]")
            stats_table = Table()
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            stats_table.add_row("Session ID", self.session_id)
            stats_table.add_row("User ID", context.user_id or "N/A")
            stats_table.add_row("Messages", str(len(context.message_history)))

            console.print(stats_table)

            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to show memory: {str(e)}[/red]")
            return False

    async def show_planning_details(self):
        """Show planning and task decomposition details"""
        console.print("\n[bold blue]🧠 Planning & Task Decomposition[/bold blue]")

        console.print(
            "\n[yellow]Enter a query to see how it would be planned:[/yellow]"
        )
        query = Prompt.ask("[cyan]Query[/cyan]")

        if not query.strip():
            console.print("[red]❌ Query cannot be empty[/red]")
            return False

        try:
            # Get planning module
            if not hasattr(self.agent_server, "planning_module"):
                console.print("[yellow]Planning module not available[/yellow]")
                return False

            # Get memory context
            context = await self.agent_server.memory_manager.get_context(
                self.session_id
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing query and creating plan...", total=None
                )

                # Create plan
                plan = await self.agent_server.planning_module.create_plan(
                    query, context
                )

                progress.update(task, description="✅ Plan created!")

            # Display plan details
            console.print("\n[bold green]📋 Execution Plan Created[/bold green]")

            plan_table = Table(title="Plan Overview")
            plan_table.add_column("Attribute", style="cyan")
            plan_table.add_column("Value", style="yellow")

            plan_table.add_row("Plan ID", plan.plan_id)
            plan_table.add_row("Total Tasks", str(len(plan.tasks)))
            plan_table.add_row("Estimated Duration", f"{plan.estimated_duration:.2f}s")
            plan_table.add_row("Priority", str(plan.priority))

            console.print(plan_table)

            # Display tasks
            console.print("\n[bold]Tasks:[/bold]")
            tasks_table = Table()
            tasks_table.add_column("#", style="cyan", width=5)
            tasks_table.add_column("Task ID", style="magenta", width=20)
            tasks_table.add_column("Type", style="yellow", width=20)
            tasks_table.add_column("Description", style="dim", width=50)

            for idx, task in enumerate(plan.tasks, 1):
                tasks_table.add_row(
                    str(idx),
                    task.get("id", "N/A"),
                    task.get("type", "N/A"),
                    task.get("description", "No description")[:50],
                )

            console.print(tasks_table)

            # Display dependencies
            if plan.dependencies:
                console.print("\n[bold]Task Dependencies:[/bold]")
                for task_id, deps in plan.dependencies.items():
                    if deps:
                        console.print(f"  • {task_id} depends on: {', '.join(deps)}")

            # Display parallel groups
            if plan.parallel_groups:
                console.print(
                    f"\n[bold]Parallel Execution Groups:[/bold] {len(plan.parallel_groups)}"
                )
                for idx, group in enumerate(plan.parallel_groups, 1):
                    console.print(f"  Group {idx}: {', '.join(group)}")

            # Display recovery strategies
            if plan.recovery_strategies:
                console.print(
                    f"\n[bold]Recovery Strategies:[/bold] {len(plan.recovery_strategies)} configured"
                )

            self.session_stats["plans_created"] += 1
            return True

        except Exception as e:
            self.print_error_details(e, "Planning")

            console.print("\n[yellow]💡 Debugging Tips:[/yellow]")
            console.print("  • Check if memory manager is initialized")
            console.print("  • Verify planning module is available")
            console.print("  • Try a simpler query first")
            console.print("  • Check option 1 (Health Check) for component status")

            return False

    async def show_conversation_history(self):
        """Show conversation history"""
        console.print("\n[bold blue]📜 Conversation History[/bold blue]")

        if not self.conversation_history:
            console.print("[yellow]No conversation history yet[/yellow]")
            return False

        for idx, interaction in enumerate(self.conversation_history, 1):
            console.print(f"\n[bold cyan]Interaction #{idx}[/bold cyan]")
            console.print(f"[bold]Timestamp:[/bold] {interaction['timestamp']}")

            console.print(f"\n[cyan]User:[/cyan] {interaction['message']}")
            console.print(f"[green]Agent:[/green] {interaction['response'][:200]}...")

            if interaction.get("metadata"):
                console.print(
                    f"[dim]Metadata: {list(interaction['metadata'].keys())}[/dim]"
                )

        return True

    async def test_rag_pipeline(self):
        """Test RAG pipeline with enhancements"""
        console.print("\n[bold blue]🔎 Testing Enhanced RAG Pipeline[/bold blue]")

        query = Prompt.ask("\n[cyan]Enter search query[/cyan]")

        if not query.strip():
            console.print("[red]❌ Query cannot be empty[/red]")
            return False

        try:
            from src.rag_pipeline.main import get_rag_pipeline

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching with enhancements...", total=None)

                rag = await get_rag_pipeline()
                result = await rag.search(
                    query=query,
                    max_results=5,
                    enable_query_reformulation=True,
                    use_adaptive_retrieval=True,
                    optimize_context=True,
                )

                progress.update(task, description="✅ Search completed!")

            # Display results
            console.print(
                f"\n[bold green]Found {len(result.get('results', []))} results[/bold green]"
            )

            # Show enhancements used
            if "enhancements_used" in result:
                enh_table = Table(title="Enhancements Applied")
                enh_table.add_column("Enhancement", style="cyan")
                enh_table.add_column("Status", style="green")

                for enh, used in result["enhancements_used"].items():
                    status = "✓ Used" if used else "✗ Not used"
                    enh_table.add_row(enh.replace("_", " ").title(), status)

                console.print(enh_table)

            # Show query reformulation
            if result.get("query_reformulation", {}).get("was_reformulated"):
                console.print("\n[bold]Query Reformulation:[/bold]")
                console.print(
                    f"  Original: {result['query_reformulation']['original_query']}"
                )
                console.print(
                    f"  Reformulated: {result['query_reformulation']['reformulated_query']}"
                )

            # Show retrieval metadata
            if "retrieval_metadata" in result:
                console.print("\n[bold]Retrieval Metadata:[/bold]")
                for key, value in result["retrieval_metadata"].items():
                    console.print(f"  {key}: {value}")

            # Show context optimization
            if result.get("context_metadata", {}).get("context_optimized"):
                console.print("\n[bold]Context Optimization:[/bold]")
                ctx = result["context_metadata"]
                console.print(f"  Compression: {ctx.get('compression_ratio', 0):.1%}")
                console.print(f"  Diversity: {ctx.get('diversity_score', 0):.2f}")
                console.print(f"  Relevance: {ctx.get('relevance_score', 0):.2f}")

            # Show top results
            console.print("\n[bold]Top Results:[/bold]")
            for i, res in enumerate(result.get("results", [])[:3], 1):
                console.print(f"\n{i}. [cyan]{res.get('content', '')[:150]}...[/cyan]")
                console.print(f"   Score: {res.get('score', 0):.3f}")

            return True

        except Exception as e:
            self.print_error_details(e, "RAG Pipeline Test")
            return False

    async def collect_user_feedback(self):
        """Collect feedback on a response"""
        console.print("\n[bold blue]👍 Collect Feedback[/bold blue]")

        if not self.conversation_history:
            console.print(
                "[yellow]No conversation history to provide feedback on[/yellow]"
            )
            return False

        # Show recent interactions
        console.print("\n[bold]Recent Interactions:[/bold]")
        for i, interaction in enumerate(self.conversation_history[-5:], 1):
            console.print(f"{i}. {interaction['message'][:50]}...")

        idx = (
            int(Prompt.ask("\n[cyan]Select interaction number[/cyan]", default="1")) - 1
        )

        if idx < 0 or idx >= len(self.conversation_history):
            console.print("[red]Invalid selection[/red]")
            return False

        interaction = self.conversation_history[idx]

        # Collect rating
        rating_str = Prompt.ask("\n[cyan]Rating (0-10)[/cyan]", default="7")
        rating = float(rating_str) / 10.0

        # Collect text feedback
        text_feedback = Prompt.ask(
            "\n[cyan]Additional feedback (optional)[/cyan]", default=""
        )

        try:
            result = await self.agent_server.collect_feedback(
                session_id=self.session_id,
                query=interaction["message"],
                response=interaction["response"],
                rating=rating,
                text_feedback=text_feedback if text_feedback else None,
                user_id=self.user_id,
            )

            if result.get("success"):
                console.print(
                    f"\n[green]✅ Feedback collected! ID: {result['feedback_id']}[/green]"
                )
                return True
            else:
                console.print(f"\n[red]❌ Failed: {result.get('error')}[/red]")
                return False

        except Exception as e:
            self.print_error_details(e, "Feedback Collection")
            return False

    async def analyze_user_feedback(self):
        """Analyze collected feedback"""
        console.print("\n[bold blue]📊 Analyze Feedback[/bold blue]")

        days = int(
            Prompt.ask("\n[cyan]Analyze feedback from last N days[/cyan]", default="7")
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing feedback...", total=None)

                result = await self.agent_server.analyze_feedback(days=days)

                progress.update(task, description="✅ Analysis complete!")

            if not result.get("success"):
                console.print(
                    f"\n[yellow]{result.get('message', result.get('error'))}[/yellow]"
                )
                return False

            analysis = result.get("analysis", {})

            # Display analysis
            console.print(
                f"\n[bold green]Analyzed {result['feedback_count']} feedback entries[/bold green]"
            )

            stats_table = Table(title="Feedback Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")

            stats_table.add_row(
                "Total Feedback", str(analysis.get("total_feedback", 0))
            )
            stats_table.add_row("Positive", str(analysis.get("positive_feedback", 0)))
            stats_table.add_row("Negative", str(analysis.get("negative_feedback", 0)))
            stats_table.add_row(
                "Average Rating", f"{analysis.get('average_rating', 0):.2f}"
            )

            console.print(stats_table)

            # Common issues
            if analysis.get("common_issues"):
                console.print("\n[bold]Common Issues:[/bold]")
                for issue in analysis["common_issues"]:
                    console.print(f"  • {issue}")

            # Suggestions
            if analysis.get("suggested_improvements"):
                console.print("\n[bold]Suggested Improvements:[/bold]")
                for suggestion in analysis["suggested_improvements"]:
                    console.print(f"  • {suggestion}")

            return True

        except Exception as e:
            self.print_error_details(e, "Feedback Analysis")
            return False

    async def run_learning_cycle(self):
        """Run learning cycle from feedback"""
        console.print("\n[bold blue]🎓 Learn from Feedback[/bold blue]")

        days = int(
            Prompt.ask(
                "\n[cyan]Learn from feedback from last N days[/cyan]", default="7"
            )
        )
        auto_apply = Confirm.ask("Auto-apply optimizations?", default=False)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Learning from feedback...", total=None)

                result = await self.agent_server.learn_from_feedback(
                    days=days, auto_apply=auto_apply
                )

                progress.update(task, description="✅ Learning complete!")

            if not result.get("success"):
                console.print(f"\n[yellow]{result.get('error')}[/yellow]")
                return False

            insights = result.get("insights", {})

            console.print(
                f"\n[bold green]Learned from {result['feedback_analyzed']} feedback entries[/bold green]"
            )

            insights_table = Table(title="Learning Insights")
            insights_table.add_column("Type", style="cyan")
            insights_table.add_column("Count", style="yellow")

            insights_table.add_row(
                "Prompt Optimizations", str(insights.get("prompt_optimizations", 0))
            )
            insights_table.add_row(
                "Strategy Adjustments", str(insights.get("strategy_adjustments", 0))
            )

            console.print(insights_table)

            # Recommendations
            if insights.get("recommendations"):
                console.print("\n[bold]Recommendations:[/bold]")
                for rec in insights["recommendations"]:
                    console.print(f"  • {rec}")

            if result.get("optimizations_applied"):
                console.print("\n[green]✅ Optimizations have been applied![/green]")
            else:
                console.print(
                    "\n[yellow]ℹ️  Optimizations generated but not applied (review required)[/yellow]"
                )

            return True

        except Exception as e:
            self.print_error_details(e, "Learning Cycle")
            return False

    async def show_feedback_statistics(self):
        """Show feedback statistics"""
        console.print("\n[bold blue]📈 Feedback Statistics[/bold blue]")

        try:
            stats = await self.agent_server.get_feedback_stats()

            if not stats.get("enabled"):
                console.print("[yellow]Feedback system not enabled[/yellow]")
                return False

            stats_table = Table(title="Feedback System Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")

            stats_table.add_row("Total Feedback", str(stats.get("total_feedback", 0)))
            stats_table.add_row(
                "Recent (7 days)", str(stats.get("recent_feedback_7d", 0))
            )

            console.print(stats_table)

            # By category
            if stats.get("feedback_by_category"):
                console.print("\n[bold]Feedback by Category:[/bold]")
                for category, count in stats["feedback_by_category"].items():
                    console.print(f"  • {category}: {count}")

            return True

        except Exception as e:
            self.print_error_details(e, "Feedback Statistics")
            return False

    async def show_statistics(self):
        """Display comprehensive system statistics"""
        console.print("\n[bold cyan]📊 System Statistics[/bold cyan]")

        # Session statistics
        session_duration = datetime.now() - self.session_stats["session_start"]
        session_table = Table(title="Current Session")
        session_table.add_column("Metric", style="cyan")
        session_table.add_column("Value", style="green")

        session_table.add_row("Session ID", self.session_id)
        session_table.add_row("Duration", str(session_duration).split(".")[0])
        session_table.add_row(
            "Messages Processed", str(self.session_stats["messages_processed"])
        )
        session_table.add_row(
            "Tools Executed", str(self.session_stats["tools_executed"])
        )
        session_table.add_row("Plans Created", str(self.session_stats["plans_created"]))

        console.print(session_table)

        # Component statistics
        if hasattr(self.agent_server, "orchestrator"):
            orch = self.agent_server.orchestrator
            if hasattr(orch, "active_executions"):
                console.print(
                    f"\n[bold]Active Executions:[/bold] {len(orch.active_executions)}"
                )
            if hasattr(orch, "execution_history"):
                total_executions = sum(len(v) for v in orch.execution_history.values())
                console.print(f"[bold]Total Executions:[/bold] {total_executions}")

        return True

    async def run_complete_demo(self):
        """Run a comprehensive demonstration of all Agent Server features"""
        console.print(
            "\n[bold green]🚀 Complete Enhanced Agent Server Demo[/bold green]"
        )
        console.print(
            "[dim]This demo showcases the full agentic RAG system with feedback learning[/dim]"
        )

        demo_steps = [
            ("System Health & Component Status", self._demo_system_health),
            ("Tool Registry & Capabilities", self._demo_tool_capabilities),
            ("Intelligent Message Processing", self._demo_message_processing),
            ("Planning & Task Orchestration", self._demo_planning_orchestration),
            ("Execution Traces & Reasoning", self._demo_execution_traces),
            ("Memory & Context Management", self._demo_memory_management),
            ("Feedback & Learning System", self._demo_feedback_system),
            ("Performance Analytics", self._demo_performance_analytics),
        ]

        demo_results = {}

        for i, (step_name, step_func) in enumerate(demo_steps, 1):
            console.print(
                f"\n[bold cyan]═══ Demo Step {i}/{len(demo_steps)}: {step_name} ═══[/bold cyan]"
            )

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Executing {step_name}...", total=None)

                    success = await step_func()
                    demo_results[step_name] = success

                    if success:
                        progress.update(
                            task, description=f"✅ {step_name} completed successfully"
                        )
                        console.print(
                            f"[green]✅ Step {i} completed successfully[/green]"
                        )
                    else:
                        progress.update(
                            task, description=f"⚠️  {step_name} completed with issues"
                        )
                        console.print(
                            f"[yellow]⚠️  Step {i} completed with issues[/yellow]"
                        )

                        if not Confirm.ask("Continue with demo?", default=True):
                            return False

            except Exception as e:
                console.print(f"[red]❌ Step {i} failed: {str(e)}[/red]")
                demo_results[step_name] = False

                if self.debug_mode:
                    self.print_error_details(e, f"Demo Step {i}")

                if not Confirm.ask("Continue despite error?", default=True):
                    return False

            if i < len(demo_steps):
                console.print("\n[dim]" + "─" * 70 + "[/dim]")
                input("Press Enter to continue to next step...")

        # Demo summary
        await self._show_demo_summary(demo_results)
        return True

    async def _demo_system_health(self):
        """Demo step: Comprehensive system health check"""
        console.print(
            "[yellow]Verifying all agent components and integrations...[/yellow]"
        )

        health_ok = await self.health_check()

        if health_ok:
            console.print("[green]🏥 All systems operational and ready[/green]")
        else:
            console.print(
                "[yellow]⚠️  Some components may have issues but demo can continue[/yellow]"
            )

        return health_ok

    async def _demo_tool_capabilities(self):
        """Demo step: Tool registry and capabilities"""
        console.print(
            "[yellow]Exploring available tools and their capabilities...[/yellow]"
        )

        tools_available = await self.list_available_tools()

        if tools_available:
            console.print(
                "[green]🔧 Tool registry loaded with multiple capabilities[/green]"
            )

            # Demonstrate tool execution with detailed logging
            console.print(
                "\n[cyan]Demonstrating tool execution with full visibility...[/cyan]"
            )

            tool_demos = [
                (
                    "knowledge_retrieval",
                    {"query": "software engineering", "max_results": 2},
                    "Knowledge Retrieval",
                ),
                (
                    "code_generation",
                    {"description": "hello world function", "language": "python"},
                    "Code Generation",
                ),
            ]

            for tool_name, params, description in tool_demos:
                console.print(f"\n[bold]Testing: {description}[/bold]")
                console.print(f"[dim]Tool: {tool_name}[/dim]")
                console.print(f"[dim]Parameters: {params}[/dim]")

                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Executing {tool_name}...", total=None
                        )

                        result = await self.agent_server.execute_tool(
                            tool_name, params, self.session_id
                        )

                        progress.update(task, description=f"✅ {tool_name} completed!")

                    if result.get("status") == "success":
                        console.print(f"[green]✅ {description} successful[/green]")

                        # Show execution time
                        exec_time = result.get("execution_time", 0)
                        console.print(f"[dim]Execution time: {exec_time:.3f}s[/dim]")

                        # Show result preview
                        if result.get("result"):
                            result_str = str(result["result"])
                            preview = (
                                result_str[:150] + "..."
                                if len(result_str) > 150
                                else result_str
                            )
                            console.print(f"[dim]Result preview: {preview}[/dim]")
                    else:
                        console.print(
                            f"[yellow]⚠️  {description} completed with status: {result.get('status')}[/yellow]"
                        )

                except Exception as e:
                    console.print(
                        f"[yellow]⚠️  {description} demo failed: {str(e)}[/yellow]"
                    )
                    if self.debug_mode:
                        self.print_error_details(e, f"{description} Tool Demo")

        return tools_available

    async def _demo_message_processing(self):
        """Demo step: Intelligent message processing"""
        console.print(
            "[yellow]Testing message processing with various query types...[/yellow]"
        )

        sample_messages = [
            ("Explain what design patterns are", "Conceptual explanation"),
            ("Generate a Python function to calculate fibonacci", "Code generation"),
            ("What are software testing best practices?", "Knowledge retrieval"),
        ]

        for message, description in sample_messages:
            console.print(f"\n[cyan]Testing: {description}[/cyan]")
            console.print(f"[dim]Message: '{message}'[/dim]")

            if Confirm.ask(f"Process this message?", default=True):
                success = await self.process_message(message)
                if not success:
                    console.print(f"[yellow]⚠️  Message processing had issues[/yellow]")

        return True

    async def _demo_planning_orchestration(self):
        """Demo step: Planning and task orchestration"""
        console.print(
            "[yellow]Demonstrating planning and task decomposition...[/yellow]"
        )

        return await self.show_planning_details()

    async def _demo_execution_traces(self):
        """Demo step: Execution traces and reasoning"""
        console.print(
            "[yellow]Showing execution traces and reasoning paths...[/yellow]"
        )

        return await self.show_execution_traces()

    async def _demo_memory_management(self):
        """Demo step: Memory and context management"""
        console.print("[yellow]Displaying conversation memory and context...[/yellow]")

        return await self.show_memory_context()

    async def _demo_feedback_system(self):
        """Demo step: Feedback and learning system"""
        console.print(
            "[yellow]Demonstrating feedback collection and learning...[/yellow]"
        )

        # Show feedback stats
        await self.show_feedback_statistics()

        console.print(
            "\n[dim]Feedback collection and learning are available through menu options 13-16[/dim]"
        )
        return True

    async def _demo_performance_analytics(self):
        """Demo step: Performance analytics"""
        console.print("[yellow]Analyzing system performance and metrics...[/yellow]")

        return await self.show_statistics()

    async def _show_demo_summary(self, demo_results: Dict[str, bool]):
        """Show comprehensive demo summary"""
        console.print(
            "\n[bold green]🎉 Complete Agent Server Demo Finished![/bold green]"
        )

        # Results table
        console.print("\n[bold]Demo Results Summary:[/bold]")
        summary_table = Table(title="Demo Step Results")
        summary_table.add_column("Step", style="cyan", width=40)
        summary_table.add_column("Status", style="green", width=20)

        for step_name, success in demo_results.items():
            status = (
                "[green]✅ Success[/green]" if success else "[yellow]⚠️  Issues[/yellow]"
            )
            summary_table.add_row(step_name, status)

        console.print(summary_table)

        # Overall statistics
        total_steps = len(demo_results)
        successful_steps = sum(1 for success in demo_results.values() if success)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0

        console.print(
            f"\n[bold]Overall Success Rate:[/bold] {success_rate:.1f}% ({successful_steps}/{total_steps} steps)"
        )

        # Session stats
        session_duration = datetime.now() - self.session_stats["session_start"]
        console.print(
            f"[bold]Demo Duration:[/bold] {str(session_duration).split('.')[0]}"
        )
        console.print(
            f"[bold]Messages Processed:[/bold] {self.session_stats['messages_processed']}"
        )
        console.print(
            f"[bold]Tools Executed:[/bold] {self.session_stats['tools_executed']}"
        )

        console.print(
            "\n[dim]Use the interactive menu to explore specific features in detail[/dim]"
        )

    def show_help(self):
        """Show help and usage examples"""
        help_text = """
[bold]Agent Server Features:[/bold]

• [green]Message Processing[/green] - Natural language understanding and response generation
• [green]Tool Execution[/green] - Dynamic tool discovery and execution
• [green]Planning & Orchestration[/green] - Hierarchical task decomposition with LangGraph
• [green]Memory Management[/green] - Intelligent conversation context with pruning
• [green]Execution Traces[/green] - Detailed workflow execution paths and reasoning
• [green]Recovery Strategies[/green] - Automatic error handling and retry logic

[bold]Key Components:[/bold]

• [cyan]LangGraph Orchestrator[/cyan] - Stateful workflow management with checkpoints
• [cyan]Planning Module[/cyan] - Advanced intent classification and task decomposition
• [cyan]Memory Manager[/cyan] - Context-aware conversation management
• [cyan]Tool Registry[/cyan] - Dynamic tool registration and lifecycle management

[bold]Capabilities:[/bold]

• Knowledge retrieval and Q&A
• Code generation with validation
• Content and document generation
• Multi-step task execution
• Parallel task processing
• Error recovery and retry logic

[bold]Tips:[/bold]

• Start with complete demo to see all features
• Use planning view to understand task decomposition
• Check execution traces for debugging
• Monitor memory context for conversation flow

[bold]Debug Mode:[/bold]

• Debug mode is [green]ENABLED[/green] by default
• Shows detailed error tracebacks
• Displays execution phase information
• Logs component initialization details
• Helps troubleshoot issues quickly
"""
        console.print(Panel(help_text, title="Help", border_style="blue"))

        # Show current debug status
        debug_status = (
            "[green]ENABLED[/green]" if self.debug_mode else "[red]DISABLED[/red]"
        )
        console.print(f"\n[bold]Current Debug Mode:[/bold] {debug_status}")

        if Confirm.ask("\nToggle debug mode?", default=False):
            self.debug_mode = not self.debug_mode
            new_status = (
                "[green]ENABLED[/green]" if self.debug_mode else "[red]DISABLED[/red]"
            )
            console.print(f"Debug mode is now: {new_status}")

    def show_main_menu(self):
        """Display the main interactive menu"""
        menu = """
[bold cyan]🎯 Integrated Agent Server Interactive CLI[/bold cyan]

[bold]Core Features:[/bold]
1.  🏥 [green]Health Check[/green]           - Verify all systems
2.  🚀 [green]Complete Demo[/green]          - Full feature demonstration
3.  💬 [green]Process Message[/green]        - Send message to agent
4.  🔧 [green]List Tools[/green]             - Show available tools
5.  ⚙️  [green]Execute Tool[/green]           - Run specific tool

[bold]Advanced Features:[/bold]
6.  🧠 [green]Show Planning[/green]          - View task decomposition
7.  🔍 [green]Execution Traces[/green]       - Show reasoning & workflow
8.  💾 [green]Memory Context[/green]         - View conversation memory
9.  📜 [green]Conversation History[/green]   - Show past interactions

[bold]RAG Enhancements:[/bold]
10. 🔎 [green]Test RAG Pipeline[/green]      - Test integrated RAG search
11. 🎯 [green]Adaptive Retrieval[/green]     - Test adaptive strategies
12. 🧬 [green]Hybrid Embeddings[/green]      - Test embedding selection

[bold]Feedback & Learning:[/bold]
13. 👍 [green]Collect Feedback[/green]       - Provide feedback on responses
14. 📊 [green]Analyze Feedback[/green]       - View feedback analysis
15. 🎓 [green]Learn from Feedback[/green]    - Run learning cycle
16. 📈 [green]Feedback Stats[/green]         - View feedback statistics

[bold]System:[/bold]
17. 📊 [green]Statistics[/green]             - View system metrics
18. 📚 [green]Help[/green]                   - Show usage information
19. 🚪 [green]Exit[/green]                   - Quit application

"""
        console.print(Panel(menu, border_style="cyan"))
        return Prompt.ask(
            "[cyan]Enter your choice (1-19)[/cyan]",
            choices=[str(i) for i in range(1, 20)],
        )

    async def run_interactive_session(self):
        """Main interactive session loop"""
        self.print_banner()

        # Initial setup
        console.print("\n[yellow]🔧 Initial Setup[/yellow]")

        if not self.check_dependencies():
            return False

        if not self.check_redis():
            console.print(
                "[yellow]⚠️  Redis not available - some features may be limited[/yellow]"
            )
            if not Confirm.ask("Continue anyway?", default=True):
                return False

        if not await self.initialize_agent_server():
            console.print("[red]❌ Agent Server initialization failed[/red]")
            return False

        console.print(
            "\n[bold green]✅ Setup completed! Ready to use Agent Server[/bold green]"
        )

        # Main interaction loop
        while True:
            try:
                choice = self.show_main_menu()

                if choice == "1":
                    await self.health_check()
                elif choice == "2":
                    await self.run_complete_demo()
                elif choice == "3":
                    await self.process_message()
                elif choice == "4":
                    await self.list_available_tools()
                elif choice == "5":
                    await self.execute_tool()
                elif choice == "6":
                    await self.show_planning_details()
                elif choice == "7":
                    await self.show_execution_traces()
                elif choice == "8":
                    await self.show_memory_context()
                elif choice == "9":
                    await self.show_conversation_history()
                elif choice == "10":
                    await self.test_rag_pipeline()
                elif choice == "11":
                    console.print(
                        "[yellow]Adaptive retrieval is integrated into RAG pipeline (option 10)[/yellow]"
                    )
                elif choice == "12":
                    console.print(
                        "[yellow]Hybrid embeddings is integrated into RAG pipeline (option 10)[/yellow]"
                    )
                elif choice == "13":
                    await self.collect_user_feedback()
                elif choice == "14":
                    await self.analyze_user_feedback()
                elif choice == "15":
                    await self.run_learning_cycle()
                elif choice == "16":
                    await self.show_feedback_statistics()
                elif choice == "17":
                    await self.show_statistics()
                elif choice == "18":
                    self.show_help()
                elif choice == "19":
                    console.print(
                        "\n[bold blue]👋 Thanks for using Enhanced Agent Server CLI![/bold blue]"
                    )
                    break

                # Pause before next menu
                if choice != "12":
                    console.print("\n" + "=" * 70)
                    input("Press Enter to continue...")
                    console.clear()

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Session interrupted by user[/yellow]")
                if Confirm.ask("Exit the application?"):
                    break
            except Exception as e:
                self.print_error_details(e, "Main Loop")
                console.print(
                    "\n[yellow]💡 This is an unexpected error in the CLI itself[/yellow]"
                )
                console.print("[dim]Please report this issue if it persists[/dim]")
                input("Press Enter to continue...")

        # Cleanup
        if self.agent_server:
            console.print("[dim]Cleaning up resources...[/dim]")
            await self.agent_server.shutdown()

        return True


def main():
    """Main entry point"""
    cli = AgentServerInteractiveCLI()

    # Set up asyncio for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(cli.run_interactive_session())
    except KeyboardInterrupt:
        console.print("\n[yellow]Application terminated by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")
        console.print("[dim]Please check the requirements and try again[/dim]")


if __name__ == "__main__":
    main()
