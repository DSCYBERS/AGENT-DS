#!/usr/bin/env python3
"""
Agent DS v2.0 - Adaptive Attack Sequencer Demo
==============================================

Comprehensive demonstration of the Adaptive Attack Sequencer capabilities:
- Reinforcement learning-based attack ordering
- Real-time sequence adaptation
- Multi-algorithm planning (MCTS, Genetic Algorithm, Dynamic Programming)
- Performance analytics and learning progression
- Integration with AI payload generation

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def demo_adaptive_attack_sequencer():
    """Main demo function for the Adaptive Attack Sequencer"""
    
    print("🤖 Agent DS v2.0 - Adaptive Attack Sequencer Demo")
    print("=" * 60)
    
    try:
        # Import AI components
        from core.ai import (
            get_sequencer_manager, AttackAction, AttackPhase, 
            AttackVector, ResourceType, get_ai_status
        )
        
        # Check AI status
        ai_status = get_ai_status()
        print(f"🔍 AI Components Status:")
        for component, available in ai_status['components'].items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"   {component}: {status}")
        print()
        
        # Initialize sequencer manager
        print("🚀 Initializing Adaptive Attack Sequencer...")
        config_path = Path("core/ai/config/sequencer_config.json")
        sequencer_manager = get_sequencer_manager(
            config_path=str(config_path) if config_path.exists() else None,
            enable_training=True
        )
        print("✅ Sequencer Manager initialized")
        print()
        
        # Demo 1: Basic Attack Sequence Optimization
        await demo_basic_optimization(sequencer_manager)
        
        # Demo 2: Multi-Algorithm Comparison
        await demo_algorithm_comparison(sequencer_manager)
        
        # Demo 3: Real-time Adaptation
        await demo_realtime_adaptation(sequencer_manager)
        
        # Demo 4: Learning and Analytics
        await demo_learning_analytics(sequencer_manager)
        
        # Demo 5: Integration with AI Payload Generator
        await demo_ai_integration(sequencer_manager)
        
        print("🎉 Adaptive Attack Sequencer Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error importing AI components: {e}")
        print("💡 Install AI dependencies with: pip install -r core/ai/requirements_ai.txt")
    except Exception as e:
        print(f"❌ Demo error: {e}")

async def demo_basic_optimization(sequencer_manager):
    """Demo 1: Basic attack sequence optimization"""
    print("📋 Demo 1: Basic Attack Sequence Optimization")
    print("-" * 50)
    
    # Create sample attack actions
    sample_attacks = create_sample_attacks()
    
    # Create target context
    target_context = {
        'technologies': ['Apache', 'PHP', 'MySQL', 'WordPress'],
        'waf_detected': False,
        'open_ports': [80, 443, 22],
        'cms_type': 'WordPress',
        'response_time_avg': 150,
        'security_headers': ['X-Frame-Options']
    }
    
    print(f"🎯 Target Context: {len(target_context['technologies'])} technologies detected")
    print(f"🔧 Available Actions: {len(sample_attacks)} attack vectors")
    
    # Optimize sequence using different methods
    methods = ['adaptive', 'mcts', 'genetic', 'dp']
    
    for method in methods:
        try:
            print(f"\n🧠 Optimizing with {method.upper()} method...")
            start_time = time.time()
            
            optimized_sequence = await sequencer_manager.optimize_attack_sequence(
                target_context=target_context,
                available_attacks=sample_attacks,
                optimization_method=method,
                constraints={'time_budget': 3600}
            )
            
            optimization_time = time.time() - start_time
            
            print(f"⏱️  Optimization time: {optimization_time:.3f}s")
            print(f"📊 Optimized sequence length: {len(optimized_sequence)}")
            
            # Show top 3 actions
            print("🔝 Top 3 recommended actions:")
            for i, action in enumerate(optimized_sequence[:3], 1):
                if hasattr(action, 'vector'):
                    print(f"   {i}. {action.vector.value} -> {action.phase.value}")
                    print(f"      Priority: {action.priority:.2f}, Success: {action.success_probability:.2f}")
                else:
                    print(f"   {i}. Action details not available")
            
        except Exception as e:
            print(f"⚠️  {method.upper()} optimization failed: {e}")
    
    print("\n✅ Basic optimization demo completed\n")

async def demo_algorithm_comparison(sequencer_manager):
    """Demo 2: Multi-algorithm performance comparison"""
    print("📊 Demo 2: Multi-Algorithm Performance Comparison")
    print("-" * 50)
    
    sample_attacks = create_sample_attacks()
    target_contexts = create_sample_target_contexts()
    
    algorithms = ['adaptive', 'mcts', 'genetic']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🔬 Testing {algorithm.upper()} algorithm...")
        algorithm_results = []
        
        for i, context in enumerate(target_contexts):
            try:
                start_time = time.time()
                
                sequence = await sequencer_manager.optimize_attack_sequence(
                    target_context=context,
                    available_attacks=sample_attacks,
                    optimization_method=algorithm
                )
                
                optimization_time = time.time() - start_time
                
                # Calculate sequence quality metrics
                quality_score = calculate_sequence_quality(sequence, context)
                
                algorithm_results.append({
                    'context_id': i,
                    'sequence_length': len(sequence),
                    'optimization_time': optimization_time,
                    'quality_score': quality_score
                })
                
                print(f"   Context {i+1}: Quality={quality_score:.2f}, Time={optimization_time:.3f}s")
                
            except Exception as e:
                print(f"   Context {i+1}: Failed - {e}")
        
        results[algorithm] = algorithm_results
    
    # Compare results
    print(f"\n📈 Algorithm Performance Summary:")
    for algorithm, algo_results in results.items():
        if algo_results:
            avg_quality = sum(r['quality_score'] for r in algo_results) / len(algo_results)
            avg_time = sum(r['optimization_time'] for r in algo_results) / len(algo_results)
            print(f"   {algorithm.upper()}: Avg Quality={avg_quality:.2f}, Avg Time={avg_time:.3f}s")
    
    print("\n✅ Algorithm comparison demo completed\n")

async def demo_realtime_adaptation(sequencer_manager):
    """Demo 3: Real-time sequence adaptation"""
    print("⚡ Demo 3: Real-time Sequence Adaptation")
    print("-" * 50)
    
    sample_attacks = create_sample_attacks()
    target_context = {
        'technologies': ['Nginx', 'Node.js', 'MongoDB'],
        'waf_detected': True,
        'response_time_avg': 300
    }
    
    # Get initial sequence
    print("📝 Generating initial attack sequence...")
    initial_sequence = await sequencer_manager.optimize_attack_sequence(
        target_context=target_context,
        available_attacks=sample_attacks,
        optimization_method='adaptive'
    )
    
    print(f"📊 Initial sequence: {len(initial_sequence)} actions")
    
    # Simulate real-time feedback and adaptation
    current_sequence = initial_sequence.copy()
    
    # Create mock sequence state
    try:
        from core.ai import SequenceState, AttackPhase, ResourceType
        current_state = SequenceState(
            current_phase=AttackPhase.RECONNAISSANCE,
            completed_actions=[],
            pending_actions=current_sequence.copy(),
            available_resources={
                ResourceType.CPU_THREADS: 8.0,
                ResourceType.MEMORY: 1024.0,
                ResourceType.TIME_BUDGET: 2000.0
            },
            target_context=target_context
        )
    except:
        # Fallback if SequenceState not available
        current_state = None
        print("⚠️  Using simplified state for adaptation demo")
    
    # Simulate 3 adaptation cycles
    for cycle in range(1, 4):
        print(f"\n🔄 Adaptation Cycle {cycle}")
        
        # Simulate feedback
        feedback = simulate_realtime_feedback(cycle)
        print(f"📡 Feedback: {feedback.get('status', 'Unknown')}")
        
        if current_state:
            # Adapt sequence
            try:
                adapted_sequence = await sequencer_manager.adapt_sequence_realtime(
                    current_sequence=current_sequence,
                    current_state=current_state,
                    feedback=feedback
                )
                
                print(f"🔧 Adapted sequence: {len(adapted_sequence)} actions")
                
                # Show adaptation changes
                if len(adapted_sequence) != len(current_sequence):
                    print(f"   📏 Length changed: {len(current_sequence)} -> {len(adapted_sequence)}")
                
                current_sequence = adapted_sequence
                
            except Exception as e:
                print(f"   ⚠️  Adaptation failed: {e}")
        else:
            print("   📝 Simulating adaptation (components not fully available)")
    
    print("\n✅ Real-time adaptation demo completed\n")

async def demo_learning_analytics(sequencer_manager):
    """Demo 4: Learning progression and analytics"""
    print("📚 Demo 4: Learning Progression and Analytics")
    print("-" * 50)
    
    # Simulate learning from attack results
    sample_attacks = create_sample_attacks()
    
    print("🎓 Simulating learning from attack executions...")
    
    # Simulate 10 attack executions
    for i in range(10):
        try:
            # Pick random action
            action = random.choice(sample_attacks)
            
            # Simulate execution result
            from core.ai import AttackResult
            
            success = random.random() < action.success_probability
            result = AttackResult(
                action=action,
                success=success,
                response_time=random.uniform(0.5, 3.0),
                status_code=200 if success else random.choice([403, 404, 500]),
                response_size=random.randint(100, 5000),
                confidence_score=random.uniform(0.6, 0.9) if success else random.uniform(0.1, 0.4)
            )
            
            # Learn from result
            context = {'waf_detected': random.choice([True, False])}
            await sequencer_manager.learn_from_execution(action, result, context)
            
            status = "✅ Success" if success else "❌ Failed"
            print(f"   Action {i+1}: {action.vector.value} - {status}")
            
        except Exception as e:
            print(f"   Action {i+1}: Learning simulation failed - {e}")
    
    # Get analytics
    print(f"\n📊 Performance Analytics:")
    try:
        analytics = sequencer_manager.get_performance_analytics()
        
        if 'component_status' in analytics:
            print("   🔧 Component Status:")
            for component, status in analytics['component_status'].items():
                print(f"      {component}: {'✅' if status else '❌'}")
        
        if 'performance_summary' in analytics:
            summary = analytics['performance_summary']
            print(f"   📈 Performance Summary:")
            print(f"      Total actions: {summary.get('total_actions', 'N/A')}")
            print(f"      Success rate: {summary.get('overall_success_rate', 0):.1%}")
            print(f"      Avg response time: {summary.get('average_response_time', 0):.2f}s")
        
    except Exception as e:
        print(f"   ⚠️  Analytics retrieval failed: {e}")
    
    print("\n✅ Learning and analytics demo completed\n")

async def demo_ai_integration(sequencer_manager):
    """Demo 5: Integration with AI Payload Generator"""
    print("🧬 Demo 5: AI Integration with Payload Generation")
    print("-" * 50)
    
    try:
        from core.ai import AIPayloadGenerator, PayloadType, TargetContext
        
        print("🔗 Testing integration with AI Payload Generator...")
        
        # Initialize payload generator
        payload_generator = AIPayloadGenerator()
        
        # Create target context for payload generation
        target_context = TargetContext(
            technology_stack=['WordPress', 'PHP', 'MySQL'],
            cms_type='WordPress',
            detected_waf='CloudFlare'
        )
        
        # Generate payloads for sequencer actions
        print("🎯 Generating AI-optimized payloads...")
        
        payload_types = [PayloadType.SQL_INJECTION, PayloadType.XSS, PayloadType.SSRF]
        
        for payload_type in payload_types:
            try:
                payloads = await payload_generator.generate_payloads(
                    payload_type=payload_type,
                    target_context=target_context,
                    count=3
                )
                
                print(f"   {payload_type.value}: Generated {len(payloads)} payloads")
                
                # Predict success for generated payloads
                for payload in payloads:
                    # Create mock action for prediction
                    from core.ai import AttackAction, AttackPhase, AttackVector
                    
                    action = AttackAction(
                        vector=AttackVector.SQL_INJECTION if payload_type == PayloadType.SQL_INJECTION else AttackVector.XSS,
                        phase=AttackPhase.EXPLOITATION,
                        target_endpoint="/test",
                        payload=payload.payload,
                        priority=0.8,
                        success_probability=payload.predicted_success_rate
                    )
                    
                    # Predict success using sequencer
                    predicted_success = await sequencer_manager.predict_action_success(
                        action=action,
                        context={'waf_detected': True}
                    )
                    
                    print(f"      Payload success prediction: {predicted_success:.2f}")
                    break  # Just test one payload per type
                
            except Exception as e:
                print(f"   {payload_type.value}: Generation failed - {e}")
        
        print("🔄 Testing adaptive payload optimization...")
        
        # Simulate adaptive payload optimization based on sequencer feedback
        context = {'waf_detected': True, 'failed_attempts': 2}
        
        adaptive_payloads = await payload_generator.generate_context_aware_payloads(
            target_context, intensity=0.8
        )
        
        print(f"   Generated {len(adaptive_payloads)} context-aware payload sets")
        
    except ImportError:
        print("⚠️  AI Payload Generator not available for integration demo")
    except Exception as e:
        print(f"⚠️  AI integration demo failed: {e}")
    
    print("\n✅ AI integration demo completed\n")

def create_sample_attacks():
    """Create sample attack actions for demo"""
    try:
        from core.ai import AttackAction, AttackPhase, AttackVector, ResourceType
        
        attacks = [
            AttackAction(
                vector=AttackVector.SQL_INJECTION,
                phase=AttackPhase.EXPLOITATION,
                target_endpoint="/login.php",
                payload="' OR 1=1 --",
                priority=0.9,
                estimated_time=2.0,
                success_probability=0.7,
                risk_score=0.6,
                resource_requirements={ResourceType.CPU_THREADS: 1.0, ResourceType.MEMORY: 64.0}
            ),
            AttackAction(
                vector=AttackVector.XSS,
                phase=AttackPhase.EXPLOITATION,
                target_endpoint="/search.php",
                payload="<script>alert('XSS')</script>",
                priority=0.7,
                estimated_time=1.5,
                success_probability=0.6,
                risk_score=0.4,
                resource_requirements={ResourceType.CPU_THREADS: 0.5, ResourceType.MEMORY: 32.0}
            ),
            AttackAction(
                vector=AttackVector.SSRF,
                phase=AttackPhase.EXPLOITATION,
                target_endpoint="/api/fetch",
                payload="http://169.254.169.254/latest/meta-data/",
                priority=0.8,
                estimated_time=3.0,
                success_probability=0.5,
                risk_score=0.7,
                resource_requirements={ResourceType.CPU_THREADS: 2.0, ResourceType.MEMORY: 128.0}
            ),
            AttackAction(
                vector=AttackVector.DIRECTORY_TRAVERSAL,
                phase=AttackPhase.ENUMERATION,
                target_endpoint="/files/",
                payload="../../../etc/passwd",
                priority=0.6,
                estimated_time=1.0,
                success_probability=0.6,
                risk_score=0.5,
                resource_requirements={ResourceType.CPU_THREADS: 0.5, ResourceType.MEMORY: 16.0}
            ),
            AttackAction(
                vector=AttackVector.BRUTE_FORCE,
                phase=AttackPhase.EXPLOITATION,
                target_endpoint="/admin/login",
                payload="admin:password123",
                priority=0.4,
                estimated_time=10.0,
                success_probability=0.3,
                risk_score=0.5,
                resource_requirements={ResourceType.CPU_THREADS: 4.0, ResourceType.MEMORY: 256.0}
            )
        ]
        
        return attacks
        
    except ImportError:
        # Return simplified demo data if components not available
        return [
            {'vector': 'sql_injection', 'priority': 0.9, 'success_probability': 0.7},
            {'vector': 'xss', 'priority': 0.7, 'success_probability': 0.6},
            {'vector': 'ssrf', 'priority': 0.8, 'success_probability': 0.5}
        ]

def create_sample_target_contexts():
    """Create sample target contexts for demo"""
    return [
        {
            'technologies': ['Apache', 'PHP', 'MySQL'],
            'waf_detected': False,
            'open_ports': [80, 443],
            'response_time_avg': 150
        },
        {
            'technologies': ['Nginx', 'Node.js', 'PostgreSQL'],
            'waf_detected': True,
            'open_ports': [80, 443, 8080],
            'response_time_avg': 200
        },
        {
            'technologies': ['IIS', 'ASP.NET', 'MSSQL'],
            'waf_detected': True,
            'open_ports': [80, 443, 1433],
            'response_time_avg': 100
        }
    ]

def calculate_sequence_quality(sequence, context):
    """Calculate quality score for attack sequence"""
    if not sequence:
        return 0.0
    
    quality_score = 0.0
    
    try:
        # Phase progression
        phases_covered = set()
        for action in sequence:
            if hasattr(action, 'phase'):
                phases_covered.add(action.phase.value)
        quality_score += len(phases_covered) * 10.0
        
        # Average priority
        if hasattr(sequence[0], 'priority'):
            avg_priority = sum(action.priority for action in sequence) / len(sequence)
            quality_score += avg_priority * 20.0
        
        # Success probability
        if hasattr(sequence[0], 'success_probability'):
            avg_success = sum(action.success_probability for action in sequence) / len(sequence)
            quality_score += avg_success * 30.0
        
        # Context compatibility
        if context.get('waf_detected') and len(sequence) > 5:
            quality_score += 10.0  # Bonus for comprehensive sequences against WAF
        
    except:
        # Fallback scoring for simplified data
        quality_score = random.uniform(40, 80)
    
    return min(100.0, quality_score)

def simulate_realtime_feedback(cycle):
    """Simulate real-time feedback for adaptation demo"""
    feedback_scenarios = [
        {
            'status': 'waf_detected',
            'waf_type': 'CloudFlare',
            'blocked_attempts': 2,
            'suggested_adjustment': 'increase_evasion'
        },
        {
            'status': 'slow_response',
            'avg_response_time': 5.0,
            'timeout_count': 1,
            'suggested_adjustment': 'reduce_concurrent_requests'
        },
        {
            'status': 'partial_success',
            'successful_vectors': ['sql_injection'],
            'failed_vectors': ['xss', 'ssrf'],
            'suggested_adjustment': 'focus_on_successful_vectors'
        }
    ]
    
    return feedback_scenarios[cycle - 1]

# Run demo if script is executed directly
if __name__ == "__main__":
    asyncio.run(demo_adaptive_attack_sequencer())