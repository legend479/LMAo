#!/usr/bin/env python3
"""
Debug script to see what LangGraph actually streams
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rich.console import Console
    from rich.panel import Panel
    import json
except ImportError:
    print("Installing rich...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    import json

console = Console()


async def debug_stream():
    """Debug what LangGraph streams"""

    console.print("\n[bold cyan]🔍 LangGraph Stream Debugger[/bold cyan]\n")

    try:
        from src.agent_server.main import AgentServer

        console.print("[yellow]Initializing Agent Server...[/yellow]")
        agent_server = AgentServer()
        await agent_server.initialize()

        console.print("[green]✅ Agent Server initialized[/green]\n")

        # Create a simple test message
        message = "Generate a Python function to calculate fibonacci numbers"
        session_id = "debug_session"
        user_id = "debug_user"

        console.print(f"[bold]Test Message:[/bold] {message}\n")

        # Get the orchestrator
        orchestrator = agent_server.orchestrator

        # Create a plan
        console.print("[yellow]Creating execution plan...[/yellow]")
        from src.agent_server.planning import ConversationContext

        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            message_history=[],
            user_preferences={},
            current_topic=None,
            domain_context={},
            conversation_state={},
        )

        plan = await agent_server.planning_module.create_plan(message, context)
        console.print(f"[green]✅ Plan created: {plan.plan_id}[/green]")
        console.print(f"   Tasks: {len(plan.tasks)}\n")

        # Create workflow
        console.print("[yellow]Creating workflow graph...[/yellow]")
        workflow = await orchestrator.create_workflow_graph(plan)
        console.print("[green]✅ Workflow created[/green]\n")

        # Create initial state
        from dataclasses import asdict
        from src.agent_server.orchestrator import WorkflowState

        initial_state = WorkflowState(
            session_id=session_id, plan_id=plan.plan_id, context={"test": True}
        )

        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={
                "thread_id": f"{session_id}_{plan.plan_id}",
                "checkpoint_ns": f"debug_{session_id}",
            }
        )

        # Stream and debug
        console.print("[bold cyan]📡 Streaming workflow execution...[/bold cyan]\n")

        stream_count = 0
        async for state_update in workflow.astream(asdict(initial_state), config):
            stream_count += 1

            console.print(f"[bold yellow]Stream #{stream_count}:[/bold yellow]")
            console.print(f"[dim]Type: {type(state_update).__name__}[/dim]")

            if isinstance(state_update, dict):
                console.print(f"[dim]Keys: {list(state_update.keys())}[/dim]")

                # Show structure
                for key, value in state_update.items():
                    console.print(f"\n[cyan]  {key}:[/cyan]")

                    if isinstance(value, dict):
                        console.print(
                            f"    [dim]Type: dict with {len(value)} keys[/dim]"
                        )
                        console.print(f"    [dim]Keys: {list(value.keys())}[/dim]")

                        # Show important fields
                        if "execution_path" in value:
                            console.print(
                                f"    [green]execution_path: {value['execution_path']}[/green]"
                            )
                        if "task_results" in value:
                            console.print(
                                f"    [green]task_results: {list(value['task_results'].keys())}[/green]"
                            )
                        if "completed_tasks" in value:
                            console.print(
                                f"    [green]completed_tasks: {value['completed_tasks']}[/green]"
                            )
                    else:
                        console.print(f"    [dim]Type: {type(value).__name__}[/dim]")
            else:
                console.print(f"[red]Unexpected type: {type(state_update)}[/red]")

            console.print()

        console.print(f"[bold green]✅ Streaming complete![/bold green]")
        console.print(f"[green]Total streams: {stream_count}[/green]\n")

        # Cleanup
        await agent_server.shutdown()

    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")


def main():
    """Main entry point"""
    console.print(
        "\n[bold cyan]╔═══════════════════════════════════════════════════════════════╗[/bold cyan]"
    )
    console.print(
        "[bold cyan]║           LANGGRAPH STREAM STRUCTURE DEBUGGER                 ║[/bold cyan]"
    )
    console.print(
        "[bold cyan]╚═══════════════════════════════════════════════════════════════╝[/bold cyan]"
    )

    asyncio.run(debug_stream())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")
