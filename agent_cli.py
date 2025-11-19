#!/usr/bin/env python3
"""
Production Agent Server CLI
Clean, professional interface with full transparency
Version: 1.0 Production
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
except ImportError:
    print("Installing required packages...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout

console = Console()


class ProductionAgentCLI:
    """Production-ready Agent Server CLI with clean interface"""

    def __init__(self):
        self.agent_server = None
        self.session_id = f"session_{int(time.time())}"
        self.stats = {"messages": 0, "tools_used": 0, "start_time": datetime.now()}

    def show_banner(self):
        """Clean professional banner"""
        banner = Panel(
            "[bold cyan]Agent Server - Production CLI[/bold cyan]\n"
            "[dim]Intelligent orchestration | Multi-tool execution | Full transparency[/dim]",
            border_style="cyan",
        )
        console.print(banner)

    async def initialize(self):
        """Initialize agent server"""
        console.print("\n[cyan]Initializing Agent Server...[/cyan]")

        try:
            from src.agent_server.main import AgentServer

            self.agent_server = AgentServer()
            await self.agent_server.initialize()

            console.print("[green]✓[/green] Agent Server ready\n")
            return True
        except Exception as e:
            console.print(f"[red]✗[/red] Initialization failed: {str(e)}\n")
            return False

    async def check_system_health(self):
        """Quick system health check"""
        console.print("\n[bold]System Status[/bold]")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="green")

        # Check components
        checks = [
            (
                "Agent Server",
                self.agent_server._initialized if self.agent_server else False,
            ),
            (
                "Orchestrator",
                hasattr(self.agent_server, "orchestrator")
                and self.agent_server.orchestrator._initialized,
            ),
            (
                "Planning",
                hasattr(self.agent_server, "planning_module")
                and self.agent_server.planning_module._initialized,
            ),
            (
                "Memory",
                hasattr(self.agent_server, "memory_manager")
                and self.agent_server.memory_manager._initialized,
            ),
            ("Tools", hasattr(self.agent_server, "tool_registry")),
        ]

        all_ok = True
        for name, status in checks:
            icon = "✓" if status else "✗"
            color = "green" if status else "red"
            table.add_row(f"{name}", f"[{color}]{icon}[/{color}]")
            if not status:
                all_ok = False

        # Check RAG server
        try:
            from src.shared.services import get_rag_client

            rag_client = await asyncio.wait_for(get_rag_client(), timeout=2.0)
            await asyncio.wait_for(
                rag_client.search(query="test", max_results=1), timeout=2.0
            )
            table.add_row("RAG Server", "[green]✓[/green]")
        except:
            table.add_row("RAG Server", "[yellow]○[/yellow] (optional)")

        console.print(table)

        if all_ok:
            console.print("\n[green]All systems operational[/green]")
        else:
            console.print("\n[yellow]Some components unavailable[/yellow]")

        return all_ok

    async def process_message(self):
        """Process user message with full transparency"""
        console.print("\n[bold cyan]Message Processing[/bold cyan]")

        message = Prompt.ask("[cyan]Your message[/cyan]")
        if not message.strip():
            console.print("[yellow]Message cannot be empty[/yellow]")
            return

        console.print(f"\n[dim]Processing: {message}[/dim]")

        start_time = time.time()

        try:
            # Show processing stages
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing intent...", total=None)
                await asyncio.sleep(0.2)

                progress.update(task, description="Creating plan...")
                await asyncio.sleep(0.2)

                progress.update(task, description="Executing...")
                result = await self.agent_server.process_message(
                    message, self.session_id
                )

                progress.update(task, description="Complete")

            elapsed = time.time() - start_time

            # Show response
            console.print(f"\n[bold]Response:[/bold]")
            console.print(Panel(result["response"], border_style="green"))

            # Show execution details
            console.print(f"\n[dim]Execution time: {elapsed:.2f}s[/dim]")

            if result.get("metadata"):
                meta = result["metadata"]
                if meta.get("tasks_completed"):
                    console.print(
                        f"[dim]Tasks completed: {meta['tasks_completed']}/{meta.get('tasks_planned', '?')}[/dim]"
                    )

            self.stats["messages"] += 1

        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")

    async def list_tools(self):
        """Show available tools"""
        console.print("\n[bold]Available Tools[/bold]")

        try:
            tools = await self.agent_server.get_available_tools()
            tool_list = tools.get("tools", [])

            if not tool_list:
                console.print("[yellow]No tools registered[/yellow]")
                return

            table = Table(show_header=True)
            table.add_column("Tool", style="cyan", width=25)
            table.add_column("Description", style="dim", width=50)

            for tool in tool_list:
                table.add_row(
                    tool.get("name", "Unknown"),
                    tool.get("description", "No description")[:50],
                )

            console.print(table)
            console.print(f"\n[dim]{len(tool_list)} tools available[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

    async def show_execution_trace(self):
        """Show last execution trace"""
        console.print("\n[bold]Execution Trace[/bold]")

        if not hasattr(self.agent_server, "orchestrator"):
            console.print("[yellow]No execution history available[/yellow]")
            return

        history = getattr(self.agent_server.orchestrator, "execution_history", {})

        if self.session_id not in history or not history[self.session_id]:
            console.print("[yellow]No executions in this session[/yellow]")
            return

        last_exec = history[self.session_id][-1]

        # Show execution path
        if last_exec.execution_path:
            console.print("\n[cyan]Execution Path:[/cyan]")
            for i, step in enumerate(last_exec.execution_path, 1):
                console.print(f"  {i}. {step}")

        # Show results
        if last_exec.tool_results:
            console.print(
                f"\n[cyan]Tool Results:[/cyan] {len(last_exec.tool_results)} tools executed"
            )

        console.print(f"\n[dim]Execution time: {last_exec.execution_time:.2f}s[/dim]")
        console.print(f"[dim]Status: {last_exec.state.value}[/dim]")

    async def show_statistics(self):
        """Show session statistics"""
        console.print("\n[bold]Session Statistics[/bold]")

        duration = datetime.now() - self.stats["start_time"]

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="green")

        table.add_row("Session ID", self.session_id[:16] + "...")
        table.add_row("Duration", str(duration).split(".")[0])
        table.add_row("Messages Processed", str(self.stats["messages"]))
        table.add_row("Tools Used", str(self.stats["tools_used"]))

        console.print(table)

    def show_menu(self):
        """Display clean main menu"""
        menu = """
[bold cyan]Agent Server CLI[/bold cyan]

[bold]Core Operations[/bold]
  1. Process Message      - Send message to agent
  2. List Tools           - Show available tools
  3. System Health        - Check component status
  4. Execution Trace      - View last execution details
  5. Statistics           - Session statistics
  6. Help                 - Usage information
  7. Exit                 - Quit application

"""
        console.print(Panel(menu, border_style="cyan"))

        # Show quick stats
        duration = datetime.now() - self.stats["start_time"]
        console.print(
            f"[dim]Session: {self.session_id[:12]}... | "
            f"Duration: {str(duration).split('.')[0]} | "
            f"Messages: {self.stats['messages']}[/dim]\n"
        )

        return Prompt.ask(
            "[cyan]Choice[/cyan]", choices=["1", "2", "3", "4", "5", "6", "7"]
        )

    def show_help(self):
        """Show usage help"""
        help_text = """
[bold cyan]Agent Server CLI - Help[/bold cyan]

[bold]Quick Start:[/bold]
  1. Check system health (Option 3)
  2. Send a message (Option 1)
  3. View execution trace (Option 4)

[bold]Features:[/bold]
  • Multi-tool orchestration
  • Intelligent task planning
  • Conversation memory
  • Full execution transparency

[bold]Tips:[/bold]
  • Start with simple queries
  • Check health if errors occur
  • View trace to understand execution
  • RAG server optional but recommended

[bold]RAG Server:[/bold]
  Start: python -m src.rag_pipeline.main
  Port: 8001 (default)
  Status: Check in System Health

[bold]Support:[/bold]
  • Check logs for detailed errors
  • View execution trace for debugging
  • System health shows component status
"""
        console.print(Panel(help_text, border_style="cyan"))

    async def run(self):
        """Main application loop"""
        self.show_banner()

        # Initialize
        if not await self.initialize():
            console.print(
                "[red]Failed to start. Check configuration and try again.[/red]"
            )
            return

        # Quick health check
        await self.check_system_health()

        # Main loop
        while True:
            try:
                choice = self.show_menu()

                if choice == "1":
                    await self.process_message()
                elif choice == "2":
                    await self.list_tools()
                elif choice == "3":
                    await self.check_system_health()
                elif choice == "4":
                    await self.show_execution_trace()
                elif choice == "5":
                    await self.show_statistics()
                elif choice == "6":
                    self.show_help()
                elif choice == "7":
                    console.print("\n[cyan]Shutting down...[/cyan]")
                    if self.agent_server:
                        await self.agent_server.shutdown()
                    console.print("[green]Goodbye![/green]\n")
                    break

                # Pause
                if choice != "7":
                    input("\nPress Enter to continue...")
                    console.clear()

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted[/yellow]")
                if Confirm.ask("Exit?"):
                    break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                input("Press Enter to continue...")


def main():
    """Entry point"""
    cli = ProductionAgentCLI()

    # Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Terminated[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")


if __name__ == "__main__":
    main()
