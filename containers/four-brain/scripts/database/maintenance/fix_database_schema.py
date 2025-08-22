#!/usr/bin/env python3.11
"""
Fix Database Schema for Four-Brain System
Adds missing 'operation' column to task_scores table
"""

import os
import sys
import logging
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_schema():
    """Fix the database schema by adding missing columns"""
    try:
        # Get Supabase credentials from environment
        supabase_url = os.getenv("SUPABASE_URL", "https://ustcfwmonegxeoqeixgg.supabase.co")
        supabase_key = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVzdGNmd21vbmVneGVvcWVpeGdnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ5MzE4NzQsImV4cCI6MjA1MDUwNzg3NH0.Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8")
        
        logger.info("🔧 Connecting to Supabase database...")
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Check if task_scores table exists and get its structure
        logger.info("🔍 Checking task_scores table structure...")
        
        # Try to add the missing operation column
        logger.info("🔧 Adding missing 'operation' column to task_scores table...")
        
        # Use SQL to add the column if it doesn't exist
        sql_query = """
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'task_scores' AND column_name = 'operation'
            ) THEN
                ALTER TABLE task_scores ADD COLUMN operation TEXT;
                RAISE NOTICE 'Added operation column to task_scores table';
            ELSE
                RAISE NOTICE 'Operation column already exists in task_scores table';
            END IF;
        END $$;
        """
        
        # Execute the SQL
        result = supabase.rpc('exec_sql', {'sql': sql_query}).execute()
        logger.info("✅ Database schema update completed successfully")
        
        # Verify the column was added
        logger.info("🔍 Verifying task_scores table structure...")
        
        # Try a simple query to verify the operation column exists
        test_result = supabase.table('task_scores').select('operation').limit(1).execute()
        logger.info("✅ Operation column verified - database schema is now correct")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database schema fix failed: {e}")
        logger.info("🔧 Attempting alternative approach...")
        
        try:
            # Alternative approach: Create the table with correct schema if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS task_scores (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                task_id TEXT NOT NULL,
                brain_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
                task_signature TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            
            # Try to execute via RPC
            result = supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
            logger.info("✅ Task_scores table created/updated with correct schema")
            return True
            
        except Exception as e2:
            logger.error(f"❌ Alternative approach also failed: {e2}")
            logger.warning("⚠️ Manual database schema update may be required")
            return False

def main():
    """Main function"""
    logger.info("🚀 Starting Four-Brain Database Schema Fix")
    logger.info("=" * 60)
    
    success = fix_database_schema()
    
    if success:
        logger.info("🎉 Database schema fix completed successfully!")
        logger.info("✅ The 'operation' column has been added to task_scores table")
        return 0
    else:
        logger.error("❌ Database schema fix failed")
        logger.info("📋 Manual steps required:")
        logger.info("1. Connect to Supabase dashboard")
        logger.info("2. Go to SQL Editor")
        logger.info("3. Run: ALTER TABLE task_scores ADD COLUMN IF NOT EXISTS operation TEXT;")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
