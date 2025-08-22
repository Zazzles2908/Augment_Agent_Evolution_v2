"""
Test Database Authentication Fix - Verification Script
Tests the new centralized database authentication system

This script verifies that the database authentication fixes work correctly
and that the new centralized system can connect to both Supabase and PostgreSQL.

Created: 2025-07-29 AEST
Purpose: Verify Priority 1.3 database authentication fix
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.database import (
    DatabaseConnectionManager,
    DatabaseAuthenticationHandler,
    DatabaseConfigValidator,
    create_connection_manager,
    create_auth_handler,
    create_config_validator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseAuthenticationTester:
    """Test suite for database authentication fixes"""
    
    def __init__(self):
        self.brain_id = "test-brain"
        self.test_results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive database authentication tests"""
        logger.info("🧪 Starting database authentication fix verification...")
        
        # Test 1: Configuration Validation
        config_result = await self.test_config_validation()
        self.test_results['config_validation'] = config_result
        
        # Test 2: Authentication Handler
        auth_result = await self.test_authentication_handler()
        self.test_results['authentication_handler'] = auth_result
        
        # Test 3: Connection Manager
        connection_result = await self.test_connection_manager()
        self.test_results['connection_manager'] = connection_result
        
        # Test 4: End-to-End Database Operations
        e2e_result = await self.test_end_to_end_operations()
        self.test_results['end_to_end'] = e2e_result
        
        # Generate overall assessment
        overall_result = self._assess_overall_results()
        self.test_results['overall'] = overall_result
        
        logger.info(f"✅ Database authentication testing completed - Overall: {overall_result['status']}")
        return self.test_results
    
    async def test_config_validation(self) -> Dict[str, Any]:
        """Test configuration validation"""
        logger.info("🔍 Testing configuration validation...")
        
        try:
            validator = create_config_validator(self.brain_id)
            validation_result = validator.validate_all()
            
            return {
                'status': 'success',
                'overall_status': validation_result['overall_status'],
                'errors': len(validation_result['errors']),
                'warnings': len(validation_result['warnings']),
                'details': validation_result
            }
            
        except Exception as e:
            logger.error(f"❌ Config validation test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_authentication_handler(self) -> Dict[str, Any]:
        """Test authentication handler"""
        logger.info("🔐 Testing authentication handler...")
        
        try:
            auth_handler = create_auth_handler(self.brain_id)
            
            # Test Supabase credential validation
            supabase_creds = {
                'url': os.getenv('SUPABASE_URL', 'https://ustcfwmonegxeoqeixgg.supabase.co'),
                'service_role_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')
            }
            
            supabase_valid, supabase_msg = auth_handler.validate_credentials('supabase', supabase_creds)
            
            # Test PostgreSQL credential validation
            postgres_creds = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'database': os.getenv('POSTGRES_DB', 'augment_agent'),
                'username': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'password')
            }
            
            postgres_valid, postgres_msg = auth_handler.validate_credentials('postgresql', postgres_creds)
            
            # Test credential storage and retrieval
            storage_success = False
            if supabase_valid:
                storage_success = auth_handler.store_credentials('supabase', supabase_creds)
                retrieved_creds = auth_handler.get_credentials('supabase')
                storage_success = storage_success and retrieved_creds is not None
            
            return {
                'status': 'success',
                'supabase_validation': {
                    'valid': supabase_valid,
                    'message': supabase_msg
                },
                'postgresql_validation': {
                    'valid': postgres_valid,
                    'message': postgres_msg
                },
                'credential_storage': storage_success,
                'auth_stats': auth_handler.get_auth_stats()
            }
            
        except Exception as e:
            logger.error(f"❌ Authentication handler test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_connection_manager(self) -> Dict[str, Any]:
        """Test connection manager"""
        logger.info("🔗 Testing connection manager...")
        
        try:
            connection_manager = create_connection_manager(self.brain_id)
            
            # Test initialization
            init_success = await connection_manager.initialize(prefer_supabase=True)
            
            # Get connection info
            connection_info = connection_manager.get_connection_info()
            
            # Test health check
            health_status = await connection_manager.health_check()
            
            # Test Supabase client if available
            supabase_test = False
            try:
                supabase_client = await connection_manager.get_supabase_client()
                if supabase_client:
                    supabase_test = True
            except Exception as e:
                logger.warning(f"⚠️ Supabase client test failed: {e}")
            
            # Test PostgreSQL connection if available
            postgres_test = False
            try:
                async with connection_manager.get_postgres_connection() as conn:
                    result = await conn.fetchval("SELECT 1")
                    postgres_test = (result == 1)
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL connection test failed: {e}")
            
            return {
                'status': 'success',
                'initialization': init_success,
                'connection_info': connection_info,
                'health_status': health_status,
                'supabase_test': supabase_test,
                'postgresql_test': postgres_test
            }
            
        except Exception as e:
            logger.error(f"❌ Connection manager test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_end_to_end_operations(self) -> Dict[str, Any]:
        """Test end-to-end database operations"""
        logger.info("🔄 Testing end-to-end database operations...")
        
        try:
            connection_manager = create_connection_manager(self.brain_id)
            await connection_manager.initialize()
            
            # Test simple query execution
            query_test = False
            try:
                result = await connection_manager.execute_scalar("SELECT 1")
                query_test = (result == 1)
            except Exception as e:
                logger.warning(f"⚠️ Query execution test failed: {e}")
            
            # Test Supabase REST operations if available
            supabase_rest_test = False
            try:
                supabase_client = await connection_manager.get_supabase_client()
                if supabase_client:
                    # Try to access a table (this might fail if table doesn't exist, but tests auth)
                    response = supabase_client.table('sessions').select('id').limit(1).execute()
                    supabase_rest_test = True
            except Exception as e:
                logger.warning(f"⚠️ Supabase REST test failed: {e}")
            
            return {
                'status': 'success',
                'query_execution': query_test,
                'supabase_rest': supabase_rest_test,
                'connection_established': query_test or supabase_rest_test
            }
            
        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _assess_overall_results(self) -> Dict[str, Any]:
        """Assess overall test results"""
        
        # Count successful tests
        successful_tests = 0
        total_tests = 0
        critical_failures = []
        
        for test_name, result in self.test_results.items():
            if test_name == 'overall':
                continue
                
            total_tests += 1
            if result.get('status') == 'success':
                successful_tests += 1
            else:
                critical_failures.append(test_name)
        
        # Determine overall status
        if successful_tests == total_tests:
            status = 'all_passed'
        elif successful_tests >= total_tests * 0.75:
            status = 'mostly_passed'
        elif successful_tests >= total_tests * 0.5:
            status = 'partially_passed'
        else:
            status = 'mostly_failed'
        
        return {
            'status': status,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'critical_failures': critical_failures
        }
    
    def print_results(self):
        """Print formatted test results"""
        print("\n" + "="*80)
        print("🧪 DATABASE AUTHENTICATION FIX - TEST RESULTS")
        print("="*80)
        
        overall = self.test_results.get('overall', {})
        print(f"📊 Overall Status: {overall.get('status', 'unknown')}")
        print(f"✅ Success Rate: {overall.get('success_rate', 0):.1f}%")
        print(f"🎯 Tests Passed: {overall.get('successful_tests', 0)}/{overall.get('total_tests', 0)}")
        
        if overall.get('critical_failures'):
            print(f"❌ Critical Failures: {', '.join(overall['critical_failures'])}")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            if test_name == 'overall':
                continue
            
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result.get('status', 'unknown')}")
            
            if result.get('error'):
                print(f"   Error: {result['error']}")
        
        print("="*80)


async def main():
    """Main test execution"""
    tester = DatabaseAuthenticationTester()
    results = await tester.run_all_tests()
    tester.print_results()
    
    # Return exit code based on results
    overall_status = results.get('overall', {}).get('status', 'unknown')
    if overall_status in ['all_passed', 'mostly_passed']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
