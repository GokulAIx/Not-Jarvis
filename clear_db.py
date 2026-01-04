"""
Quick script to clear all checkpointer state from database.
Run this when you want to start fresh without old conversation history.
"""
import os
import asyncio
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

load_dotenv()

async def clear_checkpoints():
    DB_URI = os.getenv("DATABASE_URL")
    
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=1,
        open=False,
        kwargs={
            "autocommit": True,
        }
    )
    
    try:
        await pool.open()
        
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Delete all checkpoint data
                await cur.execute("DELETE FROM checkpoints;")
                await cur.execute("DELETE FROM checkpoint_writes;")
                print("✅ All checkpoint data cleared!")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(clear_checkpoints())
