#!/usr/bin/env python3
"""
Four-Brain System Database Sharding Manager
Production-grade database sharding for high-scale deployments
Version: Production v1.0
"""

import os
import sys
import json
import hashlib
import logging
import asyncio
import asyncpg
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShardingStrategy(Enum):
    """Database sharding strategies"""
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"
    DIRECTORY_BASED = "directory_based"
    CONSISTENT_HASH = "consistent_hash"

@dataclass
class ShardConfig:
    """Database shard configuration"""
    shard_id: str
    host: str
    port: int
    database: str
    username: str
    password: str
    max_connections: int
    weight: float = 1.0
    is_active: bool = True
    is_read_only: bool = False

@dataclass
class ShardingRule:
    """Sharding rule definition"""
    table_name: str
    shard_key: str
    strategy: ShardingStrategy
    num_shards: int
    replication_factor: int = 1
    read_preference: str = "primary"  # primary, secondary, nearest

class DatabaseShardingManager:
    """Database sharding management system"""
    
    def __init__(self):
        self.config_file = '/app/scaling/sharding_config.json'
        self.shards = {}
        self.sharding_rules = {}
        self.connection_pools = {}
        
        # Initialize default sharding configuration
        self._initialize_default_shards()
        self._initialize_default_rules()
        
        logger.info("Database sharding manager initialized")
    
    def _initialize_default_shards(self):
        """Initialize default shard configuration"""
        # Primary shard (existing database)
        self.shards['shard_0'] = ShardConfig(
            shard_id='shard_0',
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DB', 'ai_system_prod'),
            username=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'augmentai_postgres_2024_prod'),
            max_connections=50,
            weight=1.0
        )
        
        # Additional shards for scaling
        for i in range(1, 4):  # Create 3 additional shards
            self.shards[f'shard_{i}'] = ShardConfig(
                shard_id=f'shard_{i}',
                host=f'postgres-shard-{i}',
                port=5432,
                database=f'ai_system_shard_{i}',
                username='postgres',
                password='augmentai_postgres_2024_prod',
                max_connections=50,
                weight=1.0,
                is_active=False  # Inactive by default
            )
    
    def _initialize_default_rules(self):
        """Initialize default sharding rules"""
        # Memory store sharding (by brain_id)
        self.sharding_rules['memory_scores'] = ShardingRule(
            table_name='memory_scores',
            shard_key='brain_id',
            strategy=ShardingStrategy.HASH_BASED,
            num_shards=4,
            replication_factor=2
        )
        
        # Task history sharding (by task_id)
        self.sharding_rules['task_history'] = ShardingRule(
            table_name='task_history',
            shard_key='task_id',
            strategy=ShardingStrategy.HASH_BASED,
            num_shards=4,
            replication_factor=1
        )
        
        # User data sharding (by user_id)
        self.sharding_rules['users'] = ShardingRule(
            table_name='users',
            shard_key='user_id',
            strategy=ShardingStrategy.HASH_BASED,
            num_shards=2,
            replication_factor=2
        )
        
        # Metrics sharding (by timestamp range)
        self.sharding_rules['metrics'] = ShardingRule(
            table_name='metrics',
            shard_key='timestamp',
            strategy=ShardingStrategy.RANGE_BASED,
            num_shards=4,
            replication_factor=1
        )
    
    async def initialize_connection_pools(self):
        """Initialize connection pools for all active shards"""
        for shard_id, shard_config in self.shards.items():
            if shard_config.is_active:
                try:
                    pool = await asyncpg.create_pool(
                        host=shard_config.host,
                        port=shard_config.port,
                        database=shard_config.database,
                        user=shard_config.username,
                        password=shard_config.password,
                        min_size=5,
                        max_size=shard_config.max_connections
                    )
                    self.connection_pools[shard_id] = pool
                    logger.info(f"Connection pool created for shard: {shard_id}")
                except Exception as e:
                    logger.error(f"Failed to create connection pool for {shard_id}: {e}")
    
    def get_shard_for_key(self, table_name: str, shard_key_value: Any) -> str:
        """Determine which shard to use for a given key"""
        if table_name not in self.sharding_rules:
            # Default to primary shard if no rule exists
            return 'shard_0'
        
        rule = self.sharding_rules[table_name]
        
        if rule.strategy == ShardingStrategy.HASH_BASED:
            return self._hash_based_shard(shard_key_value, rule.num_shards)
        elif rule.strategy == ShardingStrategy.RANGE_BASED:
            return self._range_based_shard(shard_key_value, rule.num_shards)
        elif rule.strategy == ShardingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_shard(shard_key_value, rule.num_shards)
        else:
            return 'shard_0'
    
    def _hash_based_shard(self, key_value: Any, num_shards: int) -> str:
        """Hash-based sharding"""
        key_str = str(key_value)
        hash_value = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
        shard_index = hash_value % num_shards
        return f'shard_{shard_index}'
    
    def _range_based_shard(self, key_value: Any, num_shards: int) -> str:
        """Range-based sharding (for timestamps)"""
        if isinstance(key_value, (int, float)):
            # Simple range partitioning
            shard_index = int(key_value) % num_shards
            return f'shard_{shard_index}'
        else:
            # Fallback to hash-based
            return self._hash_based_shard(key_value, num_shards)
    
    def _consistent_hash_shard(self, key_value: Any, num_shards: int) -> str:
        """Consistent hashing (simplified implementation)"""
        # This is a simplified version - production would use a proper consistent hash ring
        return self._hash_based_shard(key_value, num_shards)
    
    async def execute_query(self, table_name: str, query: str, 
                           shard_key_value: Any = None, params: List = None) -> List[Dict]:
        """Execute query on appropriate shard"""
        if shard_key_value is not None:
            shard_id = self.get_shard_for_key(table_name, shard_key_value)
        else:
            # If no shard key provided, use primary shard
            shard_id = 'shard_0'
        
        if shard_id not in self.connection_pools:
            logger.error(f"No connection pool for shard: {shard_id}")
            return []
        
        try:
            pool = self.connection_pools[shard_id]
            async with pool.acquire() as connection:
                if params:
                    result = await connection.fetch(query, *params)
                else:
                    result = await connection.fetch(query)
                
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error executing query on {shard_id}: {e}")
            return []
    
    async def execute_cross_shard_query(self, table_name: str, query: str, 
                                       params: List = None) -> List[Dict]:
        """Execute query across all shards and aggregate results"""
        rule = self.sharding_rules.get(table_name)
        if not rule:
            # Single shard query
            return await self.execute_query(table_name, query, params=params)
        
        # Execute on all relevant shards
        tasks = []
        for i in range(rule.num_shards):
            shard_id = f'shard_{i}'
            if shard_id in self.connection_pools:
                task = self.execute_query(table_name, query, params=params)
                tasks.append(task)
        
        # Aggregate results
        all_results = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_results.extend(result)
                else:
                    logger.error(f"Error in cross-shard query: {result}")
        
        return all_results
    
    async def insert_record(self, table_name: str, record: Dict, 
                           shard_key_value: Any) -> bool:
        """Insert record into appropriate shard"""
        shard_id = self.get_shard_for_key(table_name, shard_key_value)
        
        if shard_id not in self.connection_pools:
            logger.error(f"No connection pool for shard: {shard_id}")
            return False
        
        try:
            # Build INSERT query
            columns = list(record.keys())
            placeholders = [f'${i+1}' for i in range(len(columns))]
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """
            
            pool = self.connection_pools[shard_id]
            async with pool.acquire() as connection:
                await connection.execute(query, *record.values())
                logger.debug(f"Inserted record into {table_name} on {shard_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error inserting record into {shard_id}: {e}")
            return False
    
    async def update_record(self, table_name: str, record: Dict, 
                           where_clause: str, shard_key_value: Any) -> bool:
        """Update record in appropriate shard"""
        shard_id = self.get_shard_for_key(table_name, shard_key_value)
        
        if shard_id not in self.connection_pools:
            logger.error(f"No connection pool for shard: {shard_id}")
            return False
        
        try:
            # Build UPDATE query
            set_clauses = [f"{col} = ${i+1}" for i, col in enumerate(record.keys())]
            query = f"""
                UPDATE {table_name}
                SET {', '.join(set_clauses)}
                WHERE {where_clause}
            """
            
            pool = self.connection_pools[shard_id]
            async with pool.acquire() as connection:
                result = await connection.execute(query, *record.values())
                logger.debug(f"Updated record in {table_name} on {shard_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating record in {shard_id}: {e}")
            return False
    
    async def delete_record(self, table_name: str, where_clause: str, 
                           shard_key_value: Any, params: List = None) -> bool:
        """Delete record from appropriate shard"""
        shard_id = self.get_shard_for_key(table_name, shard_key_value)
        
        if shard_id not in self.connection_pools:
            logger.error(f"No connection pool for shard: {shard_id}")
            return False
        
        try:
            query = f"DELETE FROM {table_name} WHERE {where_clause}"
            
            pool = self.connection_pools[shard_id]
            async with pool.acquire() as connection:
                if params:
                    await connection.execute(query, *params)
                else:
                    await connection.execute(query)
                logger.debug(f"Deleted record from {table_name} on {shard_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting record from {shard_id}: {e}")
            return False
    
    def add_shard(self, shard_config: ShardConfig):
        """Add a new shard to the configuration"""
        self.shards[shard_config.shard_id] = shard_config
        logger.info(f"Added new shard: {shard_config.shard_id}")
    
    def activate_shard(self, shard_id: str):
        """Activate a shard"""
        if shard_id in self.shards:
            self.shards[shard_id].is_active = True
            logger.info(f"Activated shard: {shard_id}")
        else:
            logger.error(f"Shard not found: {shard_id}")
    
    def deactivate_shard(self, shard_id: str):
        """Deactivate a shard"""
        if shard_id in self.shards:
            self.shards[shard_id].is_active = False
            logger.info(f"Deactivated shard: {shard_id}")
        else:
            logger.error(f"Shard not found: {shard_id}")
    
    async def rebalance_shards(self):
        """Rebalance data across shards (simplified implementation)"""
        logger.info("Starting shard rebalancing...")
        
        # This is a simplified implementation
        # Production would involve more sophisticated data migration
        
        for table_name, rule in self.sharding_rules.items():
            logger.info(f"Rebalancing table: {table_name}")
            
            # Get all data from all shards
            all_data = await self.execute_cross_shard_query(
                table_name, f"SELECT * FROM {table_name}"
            )
            
            # Redistribute data based on current sharding rules
            # This would be done in batches in production
            for record in all_data:
                shard_key_value = record.get(rule.shard_key)
                if shard_key_value:
                    correct_shard = self.get_shard_for_key(table_name, shard_key_value)
                    # Move record to correct shard if needed
                    # Implementation would involve careful data migration
        
        logger.info("Shard rebalancing completed")
    
    def save_configuration(self):
        """Save sharding configuration to file"""
        config = {
            'shards': {
                shard_id: asdict(shard_config) 
                for shard_id, shard_config in self.shards.items()
            },
            'sharding_rules': {
                table_name: asdict(rule) 
                for table_name, rule in self.sharding_rules.items()
            }
        }
        
        # Convert enums to strings for JSON serialization
        for rule_data in config['sharding_rules'].values():
            rule_data['strategy'] = rule_data['strategy'].value if hasattr(rule_data['strategy'], 'value') else rule_data['strategy']
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Sharding configuration saved to {self.config_file}")

async def main():
    """Main function for testing"""
    try:
        sharding_manager = DatabaseShardingManager()
        
        # Initialize connection pools
        await sharding_manager.initialize_connection_pools()
        
        # Test shard selection
        shard_id = sharding_manager.get_shard_for_key('memory_scores', 'brain1')
        logger.info(f"Shard for brain1: {shard_id}")
        
        # Save configuration
        sharding_manager.save_configuration()
        
        logger.info("Database sharding manager test completed")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
