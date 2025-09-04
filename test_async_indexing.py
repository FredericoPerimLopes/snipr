#!/usr/bin/env python3
"""Quick test script for async indexing functionality."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.index_codebase import index_codebase, get_indexing_status, cancel_indexing_task


async def test_async_indexing():
    """Test the async indexing functionality."""
    
    # Test directory (use current project)
    test_path = str(Path(__file__).parent)
    
    print("🚀 Testing async indexing...")
    
    # Start indexing
    print("\n1. Starting indexing...")
    start_result = await index_codebase(test_path)
    start_data = json.loads(start_result)
    print(f"   Result: {start_data}")
    
    if start_data["status"] == "started":
        task_id = start_data["task_id"]
        print(f"   Task ID: {task_id}")
        
        # Monitor progress
        print("\n2. Monitoring progress...")
        for i in range(10):  # Check status 10 times
            await asyncio.sleep(1)  # Wait 1 second
            status_result = await get_indexing_status(test_path)
            status_data = json.loads(status_result)
            
            if "active_task" in status_data:
                progress = status_data["active_task"]["progress"]
                print(f"   Progress: {progress['progress_percentage']:.1f}% "
                      f"({progress['files_processed']}/{progress['total_files']} files)")
                
                if status_data["active_task"]["status"] in ["completed", "failed"]:
                    break
            else:
                print("   No active task found")
                break
    
    print("\n3. Final status...")
    final_status = await get_indexing_status(test_path)
    final_data = json.loads(final_status)
    print(f"   {final_data['message']}")


if __name__ == "__main__":
    asyncio.run(test_async_indexing())