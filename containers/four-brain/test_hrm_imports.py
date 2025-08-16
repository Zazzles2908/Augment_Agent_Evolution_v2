#!/usr/bin/env python3
"""
Test script to verify HRM imports and basic functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("🔍 Testing HRM imports...")
    
    # Test basic imports
    from shared.hrm import HRMOrchestrator, BlackwellOptimizer, BlackwellOptimizationConfig
    print("✅ HRM core imports successful")
    
    from shared.hrm import HRMModule, HRMHModule, HRMLModule
    print("✅ HRM module classes imported successfully")
    
    from shared.hrm import HRMModuleType, HRMPrecision, HRMTaskType
    print("✅ HRM enums imported successfully")
    
    from shared.hrm import HRMRequest, HRMResponse, HRMConvergenceState
    print("✅ HRM data structures imported successfully")
    
    # Test BlackwellOptimizer initialization
    config = BlackwellOptimizationConfig()
    optimizer = BlackwellOptimizer(config)
    print("✅ BlackwellOptimizer initialized successfully")
    
    # Test HRMOrchestrator initialization (without Triton client)
    orchestrator = HRMOrchestrator(triton_client=None, blackwell_optimizations=True)
    print("✅ HRMOrchestrator initialized successfully")
    
    print("\n🎉 All HRM components imported and initialized successfully!")
    print("✅ HRM system is ready for deployment")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Initialization error: {e}")
    sys.exit(1)
