#!/usr/bin/env python3
"""
Clear All Cached Data Script
Clears embeddings, tools.db, Redis memory, and conversations
"""

import os
import sys
import shutil
from pathlib import Path

try:
    from rich.console import Console
    from rich.prompt import Confirm
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("Installing rich package...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.prompt import Confirm
    from rich.panel import Panel
    from rich.table import Table

console = Console()


def get_cache_locations():
    """Get all cache file and directory locations"""
    base_dir = Path(__file__).parent

    locations = {
        "embeddings_cache": base_dir / "data" / "embeddings_cache.pkl",
        "tools_db": base_dir / "data" / "tools.db",
        "rag_embeddings": base_dir / "data" / "embeddings",
        "vector_store_data": base_dir / "data" / "vector_store",
        "conversation_logs": base_dir / "data" / "conversations",
        "checkpoints": base_dir / "data" / "checkpoints",
        "temp_files": base_dir / "data" / "temp",
    }

    return locations


def check_redis_connection():
    """Check if Redis is accessible"""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()
        return True, client
    except ImportError:
        return False, "Redis package not installed"
    except Exception as e:
        return False, str(e)


def clear_redis_data(client):
    """Clear all Redis data"""
    try:
        # Get all keys
        keys = client.keys("*")

        if not keys:
            console.print("[yellow]No Redis keys found[/yellow]")
            return 0

        # Show what will be deleted
        console.print(f"\n[yellow]Found {len(keys)} Redis keys:[/yellow]")

        # Group keys by prefix
        key_groups = {}
        for key in keys:
            prefix = key.split(":")[0] if ":" in key else "other"
            if prefix not in key_groups:
                key_groups[prefix] = []
            key_groups[prefix].append(key)

        # Display grouped keys
        for prefix, group_keys in key_groups.items():
            console.print(f"  • {prefix}: {len(group_keys)} keys")

        if Confirm.ask(
            f"\n[red]Delete all {len(keys)} Redis keys?[/red]", default=False
        ):
            deleted = 0
            for key in keys:
                client.delete(key)
                deleted += 1

            console.print(f"[green]✅ Deleted {deleted} Redis keys[/green]")
            return deleted
        else:
            console.print("[yellow]Skipped Redis cleanup[/yellow]")
            return 0

    except Exception as e:
        console.print(f"[red]❌ Error clearing Redis: {str(e)}[/red]")
        return 0


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


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def scan_cache_status(locations):
    """Scan and display current cache status"""
    console.print("\n[bold cyan]📊 Current Cache Status[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("Cache Type", style="cyan", width=25)
    table.add_column("Location", style="dim", width=40)
    table.add_column("Status", style="yellow", width=15)
    table.add_column("Size", style="green", width=15)

    total_size = 0
    exists_count = 0

    for name, path in locations.items():
        if path.exists():
            exists_count += 1
            if path.is_file():
                size = path.stat().st_size
                status = "File exists"
            else:
                size = get_dir_size(path)
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                status = f"Dir ({file_count} files)"

            total_size += size
            table.add_row(
                name,
                str(path.relative_to(Path(__file__).parent)),
                status,
                format_size(size),
            )
        else:
            table.add_row(
                name,
                str(path.relative_to(Path(__file__).parent)),
                "[dim]Not found[/dim]",
                "-",
            )

    console.print(table)
    console.print(f"\n[bold]Total cache size: {format_size(total_size)}[/bold]")
    console.print(f"[bold]Cached items: {exists_count}/{len(locations)}[/bold]")

    return exists_count, total_size


def clear_file_cache(path, name):
    """Clear a single file cache"""
    try:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                path.unlink()
                console.print(f"  ✅ Deleted {name}: {format_size(size)}")
                return True
            elif path.is_dir():
                size = get_dir_size(path)
                shutil.rmtree(path)
                console.print(f"  ✅ Deleted {name} directory: {format_size(size)}")
                return True
        else:
            console.print(f"  ⊘ {name} not found")
            return False
    except Exception as e:
        console.print(f"  ❌ Error deleting {name}: {str(e)}")
        return False


def clear_all_caches(locations, clear_redis=True):
    """Clear all cache files and directories"""
    console.print("\n[bold yellow]🧹 Clearing Caches...[/bold yellow]\n")

    cleared_count = 0

    # Clear file-based caches
    for name, path in locations.items():
        if clear_file_cache(path, name):
            cleared_count += 1

    # Clear Redis
    if clear_redis:
        redis_available, redis_client = check_redis_connection()
        if redis_available:
            console.print("\n[bold cyan]Redis Cleanup:[/bold cyan]")
            redis_keys_deleted = clear_redis_data(redis_client)
            if redis_keys_deleted > 0:
                cleared_count += 1
        else:
            console.print(f"\n[yellow]⚠️  Redis not available: {redis_client}[/yellow]")

    return cleared_count


def show_banner():
    """Show script banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    CACHE CLEANUP UTILITY                      ║
║          Clear embeddings, tools.db, Redis & more             ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def main():
    """Main entry point"""
    show_banner()

    # Get cache locations
    locations = get_cache_locations()

    # Scan current status
    exists_count, total_size = scan_cache_status(locations)

    if exists_count == 0:
        console.print(
            "\n[green]✨ No caches found - everything is already clean![/green]"
        )
        return

    # Check Redis
    console.print("\n[bold cyan]🔍 Checking Redis...[/bold cyan]")
    redis_available, redis_client = check_redis_connection()

    if redis_available:
        try:
            keys = redis_client.keys("*")
            console.print(f"[green]✅ Redis connected: {len(keys)} keys found[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Redis error: {str(e)}[/yellow]")
            redis_available = False
    else:
        console.print(f"[yellow]⚠️  Redis not available: {redis_client}[/yellow]")

    # Confirmation
    console.print("\n[bold yellow]⚠️  Warning:[/bold yellow]")
    console.print("This will delete:")
    console.print("  • All cached embeddings")
    console.print("  • Tools database")
    console.print("  • Vector store data")
    console.print("  • Conversation logs")
    console.print("  • Checkpoints")
    console.print("  • Temporary files")
    if redis_available:
        console.print("  • All Redis keys (memory, sessions, metrics)")

    console.print(
        f"\n[bold]Total data to be cleared: ~{format_size(total_size)}[/bold]"
    )

    if not Confirm.ask("\n[red bold]Proceed with cleanup?[/red bold]", default=False):
        console.print("\n[yellow]Cleanup cancelled[/yellow]")
        return

    # Clear caches
    cleared_count = clear_all_caches(locations, clear_redis=redis_available)

    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]✅ Cleanup Complete![/bold green]")
    console.print(f"[green]Cleared {cleared_count} cache locations[/green]")
    console.print("=" * 70)

    # Show final status
    console.print("\n[bold cyan]📊 Final Status[/bold cyan]\n")
    final_exists, final_size = scan_cache_status(locations)

    if final_exists == 0 and final_size == 0:
        console.print("\n[bold green]✨ All caches cleared successfully![/bold green]")
    else:
        console.print(
            f"\n[yellow]⚠️  Some items remain: {final_exists} items, {format_size(final_size)}[/yellow]"
        )

    console.print(
        "\n[dim]Note: The system will recreate these caches as needed during operation.[/dim]"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Cleanup cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {str(e)}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
