#!/usr/bin/env python3
"""Simple test to query the indexed database."""

import sqlite3

def test_database():
    """Test querying the vector database."""
    
    print("🗃️  Testing database content...")
    
    try:
        conn = sqlite3.connect('.index_cache/embeddings_vec.db')
        cursor = conn.cursor()
        
        # Get basic stats
        cursor.execute('SELECT COUNT(*) FROM embeddings_vec_metadata')
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT file_path) FROM embeddings_vec_metadata')
        total_files = cursor.fetchone()[0]
        
        print(f"📊 Database Stats:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total files: {total_files}")
        
        # Get file breakdown
        print(f"\n📁 Files by chunk count:")
        cursor.execute('''
            SELECT file_path, COUNT(*) as chunk_count 
            FROM embeddings_vec_metadata 
            GROUP BY file_path 
            ORDER BY chunk_count DESC 
            LIMIT 15
        ''')
        
        for file_path, count in cursor.fetchall():
            file_name = file_path.split('/')[-1]
            print(f"   {count:3d} chunks: {file_name}")
        
        # Test content search
        print(f"\n🔍 Sample content search (for 'async'):")
        cursor.execute('''
            SELECT file_path, semantic_type, content 
            FROM embeddings_vec_metadata 
            WHERE content LIKE '%async%' 
            LIMIT 5
        ''')
        
        for file_path, sem_type, content in cursor.fetchall():
            file_name = file_path.split('/')[-1]
            preview = content.replace('\n', ' ')[:60]
            print(f"   {file_name} ({sem_type}): {preview}...")
        
        conn.close()
        print(f"\n✅ Database query test completed!")
        
    except Exception as e:
        print(f"❌ Database error: {e}")


if __name__ == "__main__":
    test_database()