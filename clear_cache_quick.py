#!/usr/bin/env python3
"""
Quick Cache Clear - No prompts, just clears everything
"""

import os
import sys
import shutil
from pathlib import Path


def clear_all():
    """Clear all caches without prompts"""
    base_dir = Path(__file__).parent

    items_cleared = []
    items_failed = []

    # File and directory caches
    cache_paths = [
        base_dir / "data" / "embeddings_cache.pkl",
        base_dir / "data" / "tools.db",
        base_dir / "data" / "embeddings",
        base_dir / "data" / "vector_store",
        base_dir / "data" / "conversations",
        base_dir / "data" / "checkpoints",
        base_dir / "data" / "temp",
    ]

    print("🧹 Clearing file caches...")
    for path in cache_paths:
        try:
            if path.exists():
                if path.is_file():
                    path.unlink()
                    items_cleared.append(str(path.name))
                    print(f"  ✅ Deleted: {path.name}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    items_cleared.append(str(path.name))
                    print(f"  ✅ Deleted directory: {path.name}")
        except Exception as e:
            items_failed.append(f"{path.name}: {str(e)}")
            print(f"  ❌ Failed: {path.name} - {str(e)}")

    # Redis cache
    print("\n🔴 Clearing Redis...")
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()

        keys = client.keys("*")
        if keys:
            for key in keys:
                client.delete(key)
            items_cleared.append(f"Redis ({len(keys)} keys)")
            print(f"  ✅ Deleted {len(keys)} Redis keys")
        else:
            print("  ⊘ No Redis keys found")
    except ImportError:
        print("  ⚠️  Redis package not installed")
    except Exception as e:
        items_failed.append(f"Redis: {str(e)}")
        print(f"  ⚠️  Redis not available: {str(e)}")

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Cleared: {len(items_cleared)} items")
    if items_cleared:
        for item in items_cleared:
            print(f"   • {item}")

    if items_failed:
        print(f"\n❌ Failed: {len(items_failed)} items")
        for item in items_failed:
            print(f"   • {item}")

    print("=" * 60)
    print("✨ Cache cleanup complete!")


if __name__ == "__main__":
    try:
        clear_all()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
