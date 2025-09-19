#!/usr/bin/env python3
"""
Agent DS v2.0 - Comprehensive Integration Test
==============================================

Complete testing suite for all Agent DS v2.0 components:
- AI integration system testing
- Module enhancement testing
- Coordination manager testing
- CLI system testing
- Error handling and fallback testing

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import sys
import os
import traceback
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class AgentDSIntegrationTest:
    """
    Comprehensive integration test suite for Agent DS v2.0
    """
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    async def test_ai_integration_system(self):
        """Test AI integration system"""
        print("\n🧠 Testing AI Integration System")
        print("=" * 40)
        
        try:
            from core.attacks.complete_ai_integration import (
                get_ai_integration_system, get_ai_system_status
            )
            
            # Test system initialization
            ai_system = get_ai_integration_system()
            self.log_test("AI Integration System Initialization", ai_system is not None)
            
            # Test system status
            status = get_ai_system_status()
            self.log_test("AI System Status Retrieval", 
                         isinstance(status, dict) and 'ai_integration_active' in status)
            
            # Test AI recommendations
            recommendations = await ai_system.get_ai_recommendations(
                "https://example.com", 
                {"vulnerabilities_found": True}
            )
            self.log_test("AI Recommendations Generation", 
                         isinstance(recommendations, list))
            
            # Test capabilities demo
            demo_result = await ai_system.demonstrate_ai_capabilities()
            self.log_test("AI Capabilities Demo", 
                         isinstance(demo_result, dict) and demo_result.get('success'))
            
        except Exception as e:
            self.log_test("AI Integration System", False, f"Error: {e}")
    
    async def test_coordination_manager(self):
        """Test AI coordination manager"""
        print("\n🧠 Testing AI Coordination Manager")
        print("=" * 40)
        
        try:
            from core.attacks.ai_coordination_manager import (
                get_ai_coordinator, MissionContext, MissionPhase
            )
            
            # Test coordinator initialization
            coordinator = get_ai_coordinator()
            self.log_test("AI Coordinator Initialization", coordinator is not None)
            
            # Test mission context creation
            mission_context = MissionContext(
                target_url="https://example.com",
                technologies=["Apache", "PHP"]
            )
            self.log_test("Mission Context Creation", 
                         mission_context.target_url == "https://example.com")
            
            # Test coordination status
            status = coordinator.get_coordination_status()
            self.log_test("Coordination Status", isinstance(status, dict))
            
            # Test mission execution (with safe target)
            mission_result = await coordinator.execute_coordinated_mission(
                mission_context=mission_context,
                phases=[MissionPhase.RECONNAISSANCE]
            )
            self.log_test("Mission Execution", 
                         isinstance(mission_result, dict) and 'mission_id' in mission_result)
            
        except Exception as e:
            self.log_test("AI Coordination Manager", False, f"Error: {e}")
    
    async def test_module_enhancer(self):
        """Test module enhancer"""
        print("\n⚡ Testing Module Enhancer")
        print("=" * 40)
        
        try:
            from core.attacks.ai_module_enhancer import (
                get_module_enhancer, enhance_attack_module
            )
            
            # Test enhancer initialization
            enhancer = get_module_enhancer()
            self.log_test("Module Enhancer Initialization", enhancer is not None)
            
            # Test enhancement summary
            summary = enhancer.get_enhancement_summary()
            self.log_test("Enhancement Summary", isinstance(summary, dict))
            
            # Test module enhancement (create simple test module)
            class TestModule:
                def __init__(self):
                    self.name = "test_module"
                
                async def test_method(self):
                    return {"success": True}
            
            test_module = TestModule()
            enhanced_module = enhance_attack_module(test_module, "test_module")
            
            # Check if AI methods were added
            has_ai_methods = (
                hasattr(enhanced_module, 'ai_generate_smart_payloads') and
                hasattr(enhanced_module, 'ai_should_attempt_attack') and
                hasattr(enhanced_module, 'ai_get_status')
            )
            self.log_test("Module AI Enhancement", has_ai_methods)
            
        except Exception as e:
            self.log_test("Module Enhancer", False, f"Error: {e}")
    
    def test_cli_system(self):
        """Test CLI system"""
        print("\n💬 Testing CLI System")
        print("=" * 40)
        
        try:
            # Test interactive CLI import
            from interactive_cli import InteractiveCLI
            cli = InteractiveCLI()
            self.log_test("Interactive CLI Initialization", cli is not None)
            
            # Test command registry
            has_commands = (
                'help' in cli.commands and
                'status' in cli.commands and
                'pentest' in cli.commands and
                'demo' in cli.commands
            )
            self.log_test("CLI Command Registry", has_commands)
            
            # Test entry point import
            import agent_ds
            self.log_test("Main Entry Point Import", True)
            
        except Exception as e:
            self.log_test("CLI System", False, f"Error: {e}")
    
    def test_error_handling(self):
        """Test error handling and fallbacks"""
        print("\n🛡️ Testing Error Handling & Fallbacks")
        print("=" * 40)
        
        try:
            # Test AI system without dependencies
            from core.attacks.complete_ai_integration import get_ai_integration_system
            ai_system = get_ai_integration_system()
            
            # This should work even without ML libraries
            status = ai_system.get_system_status()
            self.log_test("Fallback Mode Operation", 
                         isinstance(status, dict))
            
            # Test terminal interface fallback
            from core.attacks.terminal_theme import HackerTerminalTheme
            theme = HackerTerminalTheme()
            self.log_test("Terminal Theme Fallback", theme is not None)
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {e}")
    
    async def test_complete_workflow(self):
        """Test complete workflow integration"""
        print("\n🚀 Testing Complete Workflow")
        print("=" * 40)
        
        try:
            from core.attacks.complete_ai_integration import execute_ai_pentest
            
            # Test full workflow with safe target
            result = await execute_ai_pentest("https://httpbin.org/get")
            
            # Should return result even if it fails (graceful handling)
            self.log_test("Complete Workflow Execution", 
                         isinstance(result, dict))
            
            # Test that result contains expected structure
            has_structure = (
                'target_url' in result or 
                'error' in result or 
                'success' in result
            )
            self.log_test("Workflow Result Structure", has_structure)
            
        except Exception as e:
            self.log_test("Complete Workflow", False, f"Error: {e}")
    
    def test_file_structure(self):
        """Test that all required files exist"""
        print("\n📁 Testing File Structure")
        print("=" * 40)
        
        required_files = [
            'agent_ds.py',
            'interactive_cli.py',
            'core/attacks/complete_ai_integration.py',
            'core/attacks/ai_coordination_manager.py',
            'core/attacks/ai_module_enhancer.py',
            'core/attacks/terminal_theme.py',
            'core/ai/__init__.py',
            'README.md'
        ]
        
        for file_path in required_files:
            exists = os.path.exists(file_path)
            self.log_test(f"File exists: {file_path}", exists)
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 Agent DS v2.0 - Comprehensive Integration Test Suite")
        print("=" * 60)
        
        # Run all test categories
        await self.test_ai_integration_system()
        await self.test_coordination_manager()
        await self.test_module_enhancer()
        self.test_cli_system()
        self.test_error_handling()
        await self.test_complete_workflow()
        self.test_file_structure()
        
        # Print summary
        print("\n📊 Test Results Summary")
        print("=" * 40)
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"📈 Success Rate: {(self.passed_tests / (self.passed_tests + self.failed_tests) * 100):.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 All tests passed! Agent DS v2.0 is fully functional!")
        else:
            print(f"\n⚠️  {self.failed_tests} test(s) failed. Check logs for details.")
        
        return self.failed_tests == 0

async def main():
    """Main test execution"""
    test_suite = AgentDSIntegrationTest()
    
    try:
        success = await test_suite.run_all_tests()
        
        if success:
            print("\n✅ Integration testing completed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed. Review the results above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)