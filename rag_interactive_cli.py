#!/usr/bin/env python3
"""
RAG Pipeline Interactive CLI
Complete testing and verification tool for all RAG features
"""

import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
except ImportError:
    print("Installing rich package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm

console = Console()


class RAGInteractiveCLI:
    """Complete interactive CLI for RAG pipeline testing"""

    def __init__(self):
        self.rag_pipeline = None
        self.session_stats = {
            "documents_ingested": 0,
            "searches_performed": 0,
            "session_start": datetime.now(),
        }
        self.test_documents_created = False

    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════╗
║                RAG PIPELINE INTERACTIVE CLI               ║
║           Complete Testing & Verification Tool            ║
╠═══════════════════════════════════════════════════════════╣
║  🚀 Document Ingestion | 🔍 Hybrid Search | 📊 Analytics  ║
╚═══════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold blue")

    def check_dependencies(self):
        """Check and install dependencies"""
        console.print("\n[yellow]🔍 Checking dependencies...[/yellow]")

        if sys.version_info < (3, 9):
            console.print("[red]❌ Python 3.9+ required[/red]")
            return False

        try:
            import requests
        except ImportError:
            console.print("[yellow]Installing requests...[/yellow]")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])

        console.print("[green]✅ Dependencies OK[/green]")
        return True

    def check_elasticsearch(self):
        """Check Elasticsearch connectivity"""
        try:
            import requests

            response = requests.get("http://localhost:9200", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Elasticsearch running[/green]")
                return True
            else:
                console.print(
                    f"[yellow]⚠️  Elasticsearch status: {response.status_code}[/yellow]"
                )
                return False
        except Exception as e:
            console.print(f"[red]❌ Elasticsearch not accessible: {str(e)[:50]}[/red]")
            return False

    def setup_elasticsearch(self):
        """Setup Elasticsearch with Docker"""
        console.print("\n[yellow]🐳 Setting up Elasticsearch...[/yellow]")

        if not Confirm.ask("Start Elasticsearch with Docker?"):
            console.print(
                "[yellow]Please start Elasticsearch manually on localhost:9200[/yellow]"
            )
            return False

        try:
            # Check Docker
            subprocess.run(["docker", "--version"], capture_output=True, check=True)

            # Stop existing container if any
            subprocess.run(
                ["docker", "stop", "rag-elasticsearch"],
                capture_output=True,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["docker", "rm", "rag-elasticsearch"],
                capture_output=True,
                stderr=subprocess.DEVNULL,
            )

            # Start new container
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                "rag-elasticsearch",
                "-p",
                "9200:9200",
                "-e",
                "discovery.type=single-node",
                "-e",
                "xpack.security.enabled=false",
                "elasticsearch:8.11.0",
            ]

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}")
            ) as progress:
                task = progress.add_task(
                    "Starting Elasticsearch container...", total=None
                )
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    progress.update(
                        task, description="Waiting for Elasticsearch to be ready..."
                    )

                    # Wait for readiness
                    for i in range(60):
                        if self.check_elasticsearch():
                            progress.update(task, description="✅ Elasticsearch ready!")
                            return True
                        time.sleep(1)

                    console.print("[red]❌ Elasticsearch didn't start in time[/red]")
                    return False
                else:
                    console.print(f"[red]❌ Failed to start: {result.stderr}[/red]")
                    return False

        except subprocess.CalledProcessError:
            console.print("[red]❌ Docker not available[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Setup failed: {e}[/red]")
            return False

    def create_test_documents(self):
        """Create comprehensive test documents"""
        if self.test_documents_created:
            return True

        console.print("\n[yellow]📄 Creating test documents...[/yellow]")

        test_dir = Path("test_documents")
        test_dir.mkdir(exist_ok=True)

        # Sample text document
        sample_txt = """Software Engineering Best Practices

## Code Quality
- Use meaningful variable and function names
- Keep functions small and focused
- Write comprehensive unit tests
- Follow consistent coding standards
- Document your code appropriately

## Version Control
- Use Git for version control
- Write clear commit messages
- Use branching strategies
- Tag releases appropriately

## Testing Strategy
- Unit tests for individual components
- Integration tests for system interactions
- End-to-end tests for user workflows
- Performance and security testing

## Documentation
- README files with setup instructions
- API documentation with examples
- Architecture diagrams and guides
"""

        # Python code example
        code_example = '''"""
Python Code Example - User Management System
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class User:
    """User data model"""
    id: int
    name: str
    email: str
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "is_active": self.is_active,
        }


class UserService:
    """Service for managing users"""
    
    def __init__(self):
        self.users: Dict[int, User] = {}
        self._next_id = 1

    async def create_user(self, name: str, email: str) -> User:
        """Create a new user"""
        user = User(id=self._next_id, name=name, email=email)
        self.users[user.id] = user
        self._next_id += 1
        return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    async def list_users(self) -> list:
        """List all active users"""
        return [user for user in self.users.values() if user.is_active]


async def main():
    """Demo the user service"""
    service = UserService()
    
    # Create users
    user1 = await service.create_user("Alice", "alice@example.com")
    user2 = await service.create_user("Bob", "bob@example.com")
    
    print(f"Created: {user1.name}, {user2.name}")
    
    # List users
    users = await service.list_users()
    print(f"Total users: {len(users)}")


if __name__ == "__main__":
    asyncio.run(main())
'''

        # Advanced concepts document
        advanced_concepts = """# Software Architecture Concepts

## Design Patterns
- **Singleton**: Single instance pattern
- **Factory**: Object creation pattern
- **Observer**: Event notification pattern
- **Strategy**: Algorithm selection pattern

## Microservices Architecture
- Single responsibility per service
- Decentralized data management
- Fault tolerance and resilience
- Technology diversity

## API Design
- RESTful endpoints with proper HTTP methods
- GraphQL for flexible data queries
- Consistent error handling
- API versioning strategies

## Database Design
- Normalization principles
- NoSQL patterns (document, key-value, graph)
- ACID properties
- CAP theorem considerations

## Testing Strategies
- Unit tests for components
- Integration tests for interactions
- End-to-end user journey tests
- Performance and load testing

## Performance Optimization
- Caching strategies (browser, CDN, application)
- Load balancing and scaling
- Database query optimization
- Asynchronous processing
"""

        # Write test documents
        documents = {
            "sample.txt": sample_txt,
            "code_example.py": code_example,
            "advanced_concepts.md": advanced_concepts,
        }

        for filename, content in documents.items():
            with open(test_dir / filename, "w", encoding="utf-8") as f:
                f.write(content)

        console.print(f"[green]✅ Created {len(documents)} test documents[/green]")
        self.test_documents_created = True
        return True

    def create_env_file(self):
        """Create environment configuration"""
        if os.path.exists(".env"):
            return True

        env_content = """# RAG Pipeline Configuration
ENVIRONMENT=development
DEBUG=true

# Elasticsearch
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200

# Embedding Models
GENERAL_EMBEDDING_MODEL=all-mpnet-base-v2
DOMAIN_EMBEDDING_MODEL=microsoft/graphcodebert-base

# Logging
LOG_LEVEL=INFO
"""
        with open(".env", "w") as f:
            f.write(env_content)
        console.print("[green]✅ Created .env configuration[/green]")
        return True

    async def initialize_rag_pipeline(self):
        """Initialize the RAG pipeline"""
        if self.rag_pipeline is not None:
            return True

        console.print("\n[yellow]🚀 Initializing RAG Pipeline...[/yellow]")

        try:
            from src.rag_pipeline import RAGPipeline
            from src.rag_pipeline.vector_store import ElasticsearchConfig
            from src.shared.logging import get_logger

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Loading models and connecting to services...", total=None
                )

                # Create proper Elasticsearch configuration with full URLs
                elasticsearch_config = ElasticsearchConfig(
                    hosts=["http://localhost:9200"],  # Full URL with scheme
                    timeout=30,
                    max_retries=3,
                )

                self.rag_pipeline = RAGPipeline(
                    elasticsearch_config=elasticsearch_config
                )
                await self.rag_pipeline.initialize()

                progress.update(task, description="✅ RAG Pipeline initialized!")

            console.print("[green]✅ RAG Pipeline ready[/green]")
            return True

        except Exception as e:
            console.print(f"[red]❌ RAG Pipeline initialization failed: {str(e)}[/red]")
            console.print(
                "[yellow]Make sure all dependencies are installed and Elasticsearch is running[/yellow]"
            )
            return False

    async def health_check(self):
        """Comprehensive system health check"""
        console.print("\n[bold blue]🏥 System Health Check[/bold blue]")

        if not self.rag_pipeline:
            console.print("[red]❌ RAG Pipeline not initialized[/red]")
            return False

        try:
            health = await self.rag_pipeline.health_check()

            # Show raw health data for debugging
            console.print(f"[dim]Debug - Raw health data: {health}[/dim]")

            # Create health table
            table = Table(title="Component Health Status", show_header=True)
            table.add_column("Component", style="cyan", width=20)
            table.add_column("Status", style="green", width=12)
            table.add_column("Details", style="dim", width=40)

            # Overall status
            pipeline_status = health.get("pipeline", "unknown")
            status_color = "green" if pipeline_status == "healthy" else "red"
            table.add_row(
                "Pipeline",
                f"[{status_color}]{pipeline_status}[/{status_color}]",
                health.get("error", "Overall system status"),
            )

            # Component statuses
            components = health.get("components", {})
            for component, status_info in components.items():
                if isinstance(status_info, dict):
                    status = status_info.get("status", "unknown")
                    details = status_info.get("error", "Running normally")
                else:
                    status = str(status_info)
                    details = "N/A"

                status_color = "green" if status == "healthy" else "red"
                table.add_row(
                    component.replace("_", " ").title(),
                    f"[{status_color}]{status}[/{status_color}]",
                    details[:40] + "..." if len(details) > 40 else details,
                )

            console.print(table)

            # Show detailed error if pipeline is unhealthy
            if pipeline_status != "healthy":
                console.print(f"\n[red]❌ Pipeline Status: {pipeline_status}[/red]")
                if "error" in health:
                    console.print(f"[red]Error Details: {health['error']}[/red]")

                # Show component details
                console.print("\n[yellow]Component Details:[/yellow]")
                for component, status_info in components.items():
                    if (
                        isinstance(status_info, dict)
                        and status_info.get("status") != "healthy"
                    ):
                        console.print(f"  [red]• {component}: {status_info}[/red]")

            return pipeline_status == "healthy"

        except Exception as e:
            console.print(f"[red]❌ Health check failed: {str(e)}[/red]")
            import traceback

            console.print(f"[dim]Full error: {traceback.format_exc()}[/dim]")
            return False

    async def ingest_documents(self, path: str = None):
        """Document ingestion with progress tracking"""
        if not path:
            path = Prompt.ask(
                "\n[cyan]Enter path to ingest[/cyan]", default="test_documents"
            )

        if not os.path.exists(path):
            console.print(f"[red]❌ Path not found: {path}[/red]")
            return False

        console.print(f"\n[bold green]📄 Ingesting: {path}[/bold green]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing documents...", total=None)

                if os.path.isfile(path):
                    # Single file
                    result = await self.rag_pipeline.ingest_document(path)
                    progress.update(task, description="✅ Document processed!")

                    if result.get("status") == "success":
                        # Show results
                        info_table = Table(title="Ingestion Results")
                        info_table.add_column("Metric", style="cyan")
                        info_table.add_column("Value", style="green")

                        info_table.add_row(
                            "Document ID", result.get("document_id", "N/A")[:20] + "..."
                        )
                        info_table.add_row(
                            "Chunks Created", str(result.get("chunks_processed", 0))
                        )
                        info_table.add_row(
                            "Processing Time",
                            f"{result.get('processing_time', 0):.2f}s",
                        )
                        info_table.add_row(
                            "Embeddings",
                            "✅" if result.get("embeddings_generated") else "❌",
                        )

                        console.print(info_table)
                        self.session_stats["documents_ingested"] += 1
                        return True
                    else:
                        console.print(
                            f"[red]❌ Ingestion failed: {result.get('error', 'Unknown error')}[/red]"
                        )
                        return False

                else:
                    # Directory
                    result = await self.rag_pipeline.ingest_from_directories([path])
                    progress.update(task, description="✅ Directory processed!")

                    # Show results
                    info_table = Table(title="Directory Ingestion Results")
                    info_table.add_column("Metric", style="cyan")
                    info_table.add_column("Value", style="green")

                    info_table.add_row(
                        "Files Found", str(result.get("total_files_found", 0))
                    )
                    info_table.add_row(
                        "Files Processed", str(result.get("total_files_processed", 0))
                    )
                    info_table.add_row(
                        "Successful", str(result.get("successful_ingestions", 0))
                    )
                    info_table.add_row(
                        "Failed", str(result.get("failed_ingestions", 0))
                    )
                    info_table.add_row(
                        "Processing Time", f"{result.get('processing_time', 0):.2f}s"
                    )

                    console.print(info_table)

                    # Show errors if any
                    errors = result.get("errors", [])
                    if errors:
                        console.print(
                            f"\n[yellow]⚠️  {len(errors)} errors encountered:[/yellow]"
                        )
                        for error in errors[:3]:
                            console.print(f"  [red]• {error}[/red]")
                        if len(errors) > 3:
                            console.print(
                                f"  [dim]... and {len(errors) - 3} more[/dim]"
                            )

                    self.session_stats["documents_ingested"] += result.get(
                        "successful_ingestions", 0
                    )
                    return result.get("successful_ingestions", 0) > 0

        except Exception as e:
            console.print(f"[red]❌ Ingestion failed: {str(e)}[/red]")
            return False

    async def search_documents(
        self, query: str = None, search_type: str = None, max_results: int = None
    ):
        """Interactive document search"""
        if not query:
            query = Prompt.ask("\n[cyan]Enter search query[/cyan]")

        if not query.strip():
            console.print("[red]❌ Query cannot be empty[/red]")
            return False

        if not search_type:
            search_type = Prompt.ask(
                "[cyan]Search method[/cyan]",
                choices=["hybrid", "keyword", "vector"],
                default="hybrid",
            )

        if not max_results:
            max_results = int(Prompt.ask("[cyan]Max results[/cyan]", default="10"))

        console.print(f"\n[bold blue]🔍 Searching: '{query}'[/bold blue]")
        console.print(f"[dim]Method: {search_type} | Max Results: {max_results}[/dim]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching documents...", total=None)

                result = await self.rag_pipeline.search(
                    query=query, search_type=search_type, max_results=max_results
                )

                progress.update(task, description="✅ Search completed!")

            # Display results
            total_results = result.get("total_results", 0)
            processing_time = result.get("processing_time", 0)

            console.print(
                f"\n[green]Found {total_results} results in {processing_time:.3f}s[/green]"
            )

            results = result.get("results", [])
            if results:
                for i, res in enumerate(results, 1):
                    # Content preview
                    content = res.get("content", "")
                    content_preview = (
                        content[:200] + "..." if len(content) > 200 else content
                    )

                    # Metadata
                    metadata = res.get("metadata", {})

                    # Create result panel
                    result_info = f"""[bold]Score:[/bold] {res.get('score', 0):.4f}
[bold]Document:[/bold] {metadata.get('document_title', 'Unknown')}
[bold]Type:[/bold] {res.get('chunk_type', 'text')}
[bold]Words:[/bold] {metadata.get('word_count', 'N/A')}

[bold]Content:[/bold]
{content_preview}"""

                    # Add highlights if available
                    highlights = res.get("highlights", {})
                    if highlights:
                        result_info += "\n\n[bold]Highlights:[/bold]"
                        for field, highlight_list in highlights.items():
                            for highlight in highlight_list[:2]:
                                result_info += f"\n• {highlight}"

                    console.print(
                        Panel(result_info, title=f"Result {i}", border_style="blue")
                    )

                    if i >= 5 and len(results) > 5:
                        if not Confirm.ask(
                            f"\nShow remaining {len(results) - 5} results?"
                        ):
                            break
            else:
                console.print(
                    "[yellow]No results found. Try different search terms.[/yellow]"
                )

            self.session_stats["searches_performed"] += 1
            return len(results) > 0

        except Exception as e:
            console.print(f"[red]❌ Search failed: {str(e)}[/red]")
            return False

    async def compare_search_methods(self, query: str = None):
        """Compare different search methods"""
        if not query:
            query = Prompt.ask("\n[cyan]Enter query for comparison[/cyan]")

        if not query.strip():
            console.print("[red]❌ Query cannot be empty[/red]")
            return False

        console.print(
            f"\n[bold magenta]⚖️  Comparing Search Methods: '{query}'[/bold magenta]"
        )

        search_types = ["hybrid", "keyword", "vector"]
        results_by_type = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for search_type in search_types:
                task = progress.add_task(f"Running {search_type} search...", total=None)

                try:
                    result = await self.rag_pipeline.search(
                        query=query, search_type=search_type, max_results=5
                    )
                    results_by_type[search_type] = result
                    progress.update(task, description=f"✅ {search_type} completed")
                except Exception as e:
                    console.print(f"[red]{search_type} search failed: {str(e)}[/red]")
                    results_by_type[search_type] = {
                        "results": [],
                        "processing_time": 0,
                        "total_results": 0,
                    }

        # Comparison table
        comparison_table = Table(title="Search Method Comparison")
        comparison_table.add_column("Method", style="cyan")
        comparison_table.add_column("Results", style="green")
        comparison_table.add_column("Time (s)", style="yellow")
        comparison_table.add_column("Top Score", style="magenta")

        for search_type, result in results_by_type.items():
            results = result.get("results", [])
            top_score = results[0].get("score", 0) if results else 0

            comparison_table.add_row(
                search_type.title(),
                str(result.get("total_results", 0)),
                f"{result.get('processing_time', 0):.3f}",
                f"{top_score:.4f}",
            )

        console.print(comparison_table)

        # Show top result from each method
        console.print("\n[bold]Top Result from Each Method:[/bold]")
        for search_type, result in results_by_type.items():
            results = result.get("results", [])
            if results:
                top_result = results[0]
                content_preview = top_result.get("content", "")[:100] + "..."

                console.print(
                    Panel(
                        f"[bold]Score:[/bold] {top_result.get('score', 0):.4f}\n{content_preview}",
                        title=f"{search_type.title()} - Top Result",
                        border_style="dim",
                    )
                )
            else:
                console.print(
                    Panel(
                        "No results found",
                        title=f"{search_type.title()} - No Results",
                        border_style="red",
                    )
                )

        return True

    async def show_statistics(self):
        """Display comprehensive system statistics"""
        console.print("\n[bold cyan]📊 System Statistics[/bold cyan]")

        try:
            stats = await self.rag_pipeline.get_document_stats()

            # Document collection stats
            vector_stats = stats.get("vector_store_stats", {})
            if vector_stats:
                doc_table = Table(title="Document Collection")
                doc_table.add_column("Metric", style="cyan")
                doc_table.add_column("Count", style="green")

                doc_table.add_row(
                    "Total Documents", str(vector_stats.get("total_documents", 0))
                )
                doc_table.add_row(
                    "Total Chunks", str(vector_stats.get("total_chunks", 0))
                )

                # Chunk types
                chunk_types = vector_stats.get("chunk_types", {})
                for chunk_type, count in chunk_types.items():
                    doc_table.add_row(f"  {chunk_type.title()} Chunks", str(count))

                console.print(doc_table)

            # Search performance stats
            search_stats = stats.get("search_stats", {})
            if search_stats:
                search_table = Table(title="Search Performance")
                search_table.add_column("Metric", style="cyan")
                search_table.add_column("Value", style="green")

                search_table.add_row(
                    "Total Searches", str(search_stats.get("total_searches", 0))
                )
                search_table.add_row(
                    "Avg Response Time",
                    f"{search_stats.get('avg_response_time', 0):.3f}s",
                )

                # Search method usage
                search_types = search_stats.get("search_types_used", {})
                for search_type, count in search_types.items():
                    search_table.add_row(
                        f"  {search_type.title()} Searches", str(count)
                    )

                console.print(search_table)

            # Session statistics
            session_duration = datetime.now() - self.session_stats["session_start"]
            session_table = Table(title="Current Session")
            session_table.add_column("Metric", style="cyan")
            session_table.add_column("Value", style="green")

            session_table.add_row("Duration", str(session_duration).split(".")[0])
            session_table.add_row(
                "Documents Ingested", str(self.session_stats["documents_ingested"])
            )
            session_table.add_row(
                "Searches Performed", str(self.session_stats["searches_performed"])
            )

            console.print(session_table)

            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to get statistics: {str(e)}[/red]")
            return False

    async def run_complete_demo(self):
        """Run a complete demonstration of all features"""
        console.print("\n[bold green]🚀 Complete RAG Pipeline Demo[/bold green]")

        # Step 1: Health check
        console.print("\n[bold]Step 1: System Health Check[/bold]")
        health_ok = await self.health_check()
        if not health_ok:
            console.print(
                "[yellow]⚠️  System health check shows issues, but continuing with demo...[/yellow]"
            )
            if not Confirm.ask(
                "Continue with demo despite health issues?", default=True
            ):
                return False

        # Step 2: Ensure test documents exist
        console.print("\n[bold]Step 2: Preparing Test Documents[/bold]")
        self.create_test_documents()

        # Step 3: Ingest documents
        console.print("\n[bold]Step 3: Document Ingestion[/bold]")
        if not await self.ingest_documents("test_documents"):
            console.print(
                "[yellow]⚠️  Some documents may not have been ingested[/yellow]"
            )

        # Step 4: Show statistics
        console.print("\n[bold]Step 4: System Statistics[/bold]")
        await self.show_statistics()

        # Step 5: Sample searches
        console.print("\n[bold]Step 5: Sample Searches[/bold]")
        sample_queries = [
            "software engineering best practices",
            "design patterns",
            "async functions",
            "testing strategies",
        ]

        for query in sample_queries:
            if Confirm.ask(f"Search for '{query}'?", default=True):
                await self.search_documents(query, "hybrid", 3)

        # Step 6: Search comparison
        console.print("\n[bold]Step 6: Search Method Comparison[/bold]")
        await self.compare_search_methods("code quality")

        console.print("\n[bold green]✅ Complete demo finished![/bold green]")
        console.print("[dim]Use the interactive menu to explore further[/dim]")
        return True

    def show_help(self):
        """Show help and usage examples"""
        help_text = """
[bold]Features:[/bold]
• Document ingestion (files/directories)
• Search methods: hybrid, keyword, vector
• Performance monitoring and statistics
• Automated demo of all capabilities

[bold]Search Methods:[/bold]
• [green]Hybrid[/green] - Keyword + vector fusion (recommended)
• [green]Keyword[/green] - Traditional text search
• [green]Vector[/green] - Semantic similarity search

[bold]Requirements:[/bold]
• Python 3.9+, Elasticsearch on localhost:9200

[bold]Tips:[/bold]
• Start with complete demo
• Use hybrid search for best results
• Check statistics for performance monitoring
"""
        console.print(Panel(help_text, title="Help", border_style="blue"))

    def show_main_menu(self):
        """Display the main interactive menu"""
        menu = """
[bold blue]🎯 RAG Pipeline Interactive CLI[/bold blue]

[bold]Choose an option:[/bold]

1. 🏥 [green]Health Check[/green]        - Verify system status
2. 🚀 [green]Complete Demo[/green]       - Run full demonstration  
3. 📄 [green]Ingest Documents[/green]    - Add documents to search
4. 🔍 [green]Search Documents[/green]    - Find relevant content
5. ⚖️  [green]Compare Methods[/green]     - Test search approaches
6. 📊 [green]System Statistics[/green]   - View performance metrics
7. 📚 [green]Help & Examples[/green]     - Show usage information
8. 🚪 [green]Exit[/green]               - Quit the application

"""
        console.print(Panel(menu, border_style="blue"))
        return Prompt.ask(
            "[cyan]Enter your choice (1-8)[/cyan]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8"],
        )

    async def run_interactive_session(self):
        """Main interactive session loop"""
        self.print_banner()

        # Initial setup
        console.print("\n[yellow]🔧 Initial Setup[/yellow]")

        if not self.check_dependencies():
            return False

        if not self.check_elasticsearch():
            if not self.setup_elasticsearch():
                console.print("[red]❌ Elasticsearch setup failed[/red]")
                return False

        self.create_test_documents()
        self.create_env_file()

        if not await self.initialize_rag_pipeline():
            console.print("[red]❌ RAG Pipeline initialization failed[/red]")
            return False

        console.print(
            "\n[bold green]✅ Setup completed! Ready to use RAG Pipeline[/bold green]"
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
                    await self.ingest_documents()

                elif choice == "4":
                    await self.search_documents()

                elif choice == "5":
                    await self.compare_search_methods()

                elif choice == "6":
                    await self.show_statistics()

                elif choice == "7":
                    self.show_help()

                elif choice == "8":
                    console.print(
                        "\n[bold blue]👋 Thanks for using RAG Pipeline CLI![/bold blue]"
                    )
                    break

                # Pause before next menu
                if choice != "8":
                    console.print("\n" + "=" * 60)
                    input("Press Enter to continue...")
                    console.clear()

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Session interrupted by user[/yellow]")
                if Confirm.ask("Exit the application?"):
                    break
            except Exception as e:
                console.print(f"\n[red]❌ Unexpected error: {str(e)}[/red]")
                console.print("[dim]Please report this issue if it persists[/dim]")
                input("Press Enter to continue...")

        # Cleanup
        if self.rag_pipeline:
            console.print("[dim]Cleaning up resources...[/dim]")
            await self.rag_pipeline.shutdown()

        return True


def main():
    """Main entry point"""
    cli = RAGInteractiveCLI()

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
