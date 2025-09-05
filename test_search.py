#!/usr/bin/env python3
"""Test script to search the indexed codebase."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.search_code import search_code


async def test_search():
    """Test the search functionality."""
    
    print("🔍 Testing codebase search...")
    
    # Test queries
    test_queries = [
        "indexing service",
        "vector embeddings", 
        "database operations",
        "async functions",
        "logging functionality"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Searching for: '{query}'")
        
        try:
            # Perform search (disable auto_sync to avoid triggering indexing)
            result = await search_code(
                query=query,
                codebase_path="/home/flopes/snipr",
                max_results=5,
                auto_sync=False  # Don't trigger indexing during search test
            )
            
            # Parse result
            data = json.loads(result)
            
            print(f"   Raw result: {data}")
            
            if data["status"] == "success":
                if "search_result" in data and "results" in data["search_result"]:
                    results = data["search_result"]["results"]
                    print(f"   Found {len(results)} results:")
                    
                    for j, match in enumerate(results, 1):
                        file_name = match["file_path"].split("/")[-1]
                        print(f"   {j}. {file_name}:{match['start_line']}-{match['end_line']} "
                              f"(similarity: {match.get('similarity', 'N/A')})")
                        print(f"      {match['semantic_type']}: {match['content'][:80]}...")
                else:
                    print(f"   Unexpected success format: {data}")
            else:
                print(f"   Error: {data.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"   Exception: {e}")
        
        # Small delay between searches
        await asyncio.sleep(0.5)
    
    print("\n✅ Search testing completed!")


if __name__ == "__main__":
    asyncio.run(test_search())