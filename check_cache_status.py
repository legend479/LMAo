#!/usr/bin/env python3
"""
Check Cache Status - View cache information without clearing
"""

import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("Installing rich package...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

console = Console()


def get_dir_size(path):
    """Get total size of directory in bytes"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except Exception:
        pass
    return total


def count_files(path):
    """Count files in directory recursively"""
    try:
        return sum(1 for _ in path.rglob("*") if _.is_file())
    except Exception:
        return 0


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def get_file_age(path):
    """Get file age in human readable format"""
    try:
        mtime = path.stat().st_mtime
        age = datetime.now().timestamp() - mtime

        if age < 60:
            return f"{int(age)}s ago"
        elif age < 3600:
            return f"{int(age/60)}m ago"
        elif age < 86400:
            return f"{int(age/3600)}h ago"
        else:
            return f"{int(age/86400)}d ago"
    except Exception:
        return "Unknown"


def check_redis_status():
    """Check Redis status and key count"""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()

        keys = client.keys("*")

        # Group keys by prefix
        key_groups = {}
        for key in keys:
            prefix = key.split(":")[0] if ":" in key else "other"
            if prefix not in key_groups:
                key_groups[prefix] = []
            key_groups[prefix].append(key)

        # Get memory usage
        info = client.info("memory")
        used_memory = info.get("used_memory", 0)

        return True, {
            "total_keys": len(keys),
            "key_groups": key_groups,
            "memory_used": used_memory,
            "client": client,
        }
    except ImportError:
        return False, "Redis package not installed"
    except Exception as e:
        return False, str(e)


def show_file_caches():
    """Show file-based cache status"""
    base_dir = Path(__file__).parent

    cache_locations = {
        "Embeddings Cache": base_dir / "data" / "embeddings_cache.pkl",
        "Tools Database": base_dir / "data" / "tools.db",
        "RAG Embeddings": base_dir / "data" / "embeddings",
        "Vector Store": base_dir / "data" / "vector_store",
        "Conversations": base_dir / "data" / "conversations",
        "Checkpoints": base_dir / "data" / "checkpoints",
        "Temp Files": base_dir / "data" / "temp",
    }

    console.print("\n[bold cyan]📁 File-Based Caches[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("Cache", style="cyan", width=20)
    table.add_column("Status", style="yellow", width=12)
    table.add_column("Size", style="green", width=12)
    table.add_column("Items", style="magenta", width=10)
    table.add_column("Last Modified", style="dim", width=15)

    total_size = 0
    exists_count = 0

    for name, path in cache_locations.items():
        if path.exists():
            exists_count += 1

            if path.is_file():
                size = path.stat().st_size
                items = "1 file"
                age = get_file_age(path)
            else:
                size = get_dir_size(path)
                file_count = count_files(path)
                items = f"{file_count} files"
                age = get_file_age(path) if file_count > 0 else "-"

            total_size += size
            table.add_row(name, "[green]Exists[/green]", format_size(size), items, age)
        else:
            table.add_row(name, "[dim]Not found[/dim]", "-", "-", "-")

    console.print(table)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  • Total size: [green]{format_size(total_size)}[/green]")
    console.print(
        f"  • Cached items: [yellow]{exists_count}/{len(cache_locations)}[/yellow]"
    )

    return total_size, exists_count


def show_redis_status():
    """Show Redis cache status"""
    console.print("\n[bold cyan]🔴 Redis Cache[/bold cyan]\n")

    redis_available, redis_data = check_redis_status()

    if not redis_available:
        console.print(f"[yellow]⚠️  Redis not available: {redis_data}[/yellow]")
        return 0

    # Connection status
    console.print(f"[green]✅ Connected to Redis[/green]")
    console.print(f"[dim]Memory used: {format_size(redis_data['memory_used'])}[/dim]\n")

    # Keys by category
    if redis_data["total_keys"] == 0:
        console.print("[yellow]No keys found in Redis[/yellow]")
        return 0

    table = Table(show_header=True)
    table.add_column("Category", style="cyan", width=25)
    table.add_column("Keys", style="green", width=15)
    table.add_column("Example Keys", style="dim", width=50)

    for prefix, keys in sorted(redis_data["key_groups"].items()):
        example_keys = ", ".join(keys[:3])
        if len(keys) > 3:
            example_keys += f" ... (+{len(keys)-3} more)"

        table.add_row(prefix, str(len(keys)), example_keys)

    console.print(table)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  • Total keys: [green]{redis_data['total_keys']}[/green]")
    console.print(f"  • Categories: [yellow]{len(redis_data['key_groups'])}[/yellow]")
    console.print(
        f"  • Memory: [magenta]{format_size(redis_data['memory_used'])}[/magenta]"
    )

    return redis_data["total_keys"]


def show_recommendations(file_size, file_count, redis_keys):
    """Show recommendations based on cache status"""
    console.print("\n[bold cyan]💡 Recommendations[/bold cyan]\n")

    recommendations = []

    # Check file cache size
    if file_size > 100 * 1024 * 1024:  # > 100 MB
        recommendations.append(
            (
                "⚠️  Large cache size",
                f"File caches are using {format_size(file_size)}. Consider clearing if not needed.",
            )
        )
    elif file_size > 10 * 1024 * 1024:  # > 10 MB
        recommendations.append(
            (
                "ℹ️  Moderate cache size",
                f"File caches are using {format_size(file_size)}. This is normal for active use.",
            )
        )
    else:
        recommendations.append(
            (
                "✅ Small cache size",
                f"File caches are using {format_size(file_size)}. No action needed.",
            )
        )

    # Check Redis keys
    if redis_keys > 1000:
        recommendations.append(
            (
                "⚠️  Many Redis keys",
                f"{redis_keys} keys in Redis. Consider clearing old sessions.",
            )
        )
    elif redis_keys > 100:
        recommendations.append(
            (
                "ℹ️  Moderate Redis usage",
                f"{redis_keys} keys in Redis. This is normal for active use.",
            )
        )
    elif redis_keys > 0:
        recommendations.append(
            ("✅ Low Redis usage", f"{redis_keys} keys in Redis. No action needed.")
        )
    else:
        recommendations.append(
            ("ℹ️  Empty Redis", "No keys in Redis. System will populate on first use.")
        )

    # Check if caches exist
    if file_count == 0 and redis_keys == 0:
        recommendations.append(
            ("✨ Clean state", "No caches found. System is in fresh state.")
        )

    for icon, message in recommendations:
        console.print(f"{icon} {message}")

    # Action suggestions
    console.print("\n[bold]Actions:[/bold]")
    if file_size > 50 * 1024 * 1024 or redis_keys > 500:
        console.print(
            "  • Run [cyan]python clear_cache.py[/cyan] to clear caches interactively"
        )
        console.print(
            "  • Run [cyan]python clear_cache_quick.py[/cyan] for quick cleanup"
        )
    else:
        console.print("  • No cleanup needed at this time")
        console.print(
            "  • Run [cyan]python clear_cache.py[/cyan] if you want to start fresh"
        )


def main():
    """Main entry point"""
    console.print(
        "\n[bold cyan]╔═══════════════════════════════════════════════════════════════╗[/bold cyan]"
    )
    console.print(
        "[bold cyan]║                  CACHE STATUS CHECKER                         ║[/bold cyan]"
    )
    console.print(
        "[bold cyan]║            View cache information without clearing            ║[/bold cyan]"
    )
    console.print(
        "[bold cyan]╚═══════════════════════════════════════════════════════════════╝[/bold cyan]"
    )

    # Show file caches
    file_size, file_count = show_file_caches()

    # Show Redis status
    redis_keys = show_redis_status()

    # Show recommendations
    show_recommendations(file_size, file_count, redis_keys)

    # Overall summary
    console.print("\n" + "=" * 70)
    console.print("[bold]Overall Cache Status:[/bold]")
    console.print(f"  • File caches: {format_size(file_size)} ({file_count} locations)")
    console.print(f"  • Redis keys: {redis_keys}")
    console.print(f"  • Total: ~{format_size(file_size)} + Redis memory")
    console.print("=" * 70)

    console.print(
        "\n[dim]Run 'python clear_cache.py' to clear caches interactively[/dim]"
    )
    console.print("[dim]Run 'python clear_cache_quick.py' for quick cleanup[/dim]\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
