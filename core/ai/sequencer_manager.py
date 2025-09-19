#!/usr/bin/env python3
"""
Agent DS v2.0 - Adaptive Attack Sequencer Integration
====================================================

Main integration module that ties together all sequencer components:
- Adaptive Attack Sequencer (main RL system)
- Attack Sequence Planner (MCTS, GA, DP algorithms)
- Reinforcement Learning Training (DQN, Actor-Critic)
- Real-time adaptation and learning
- Performance monitoring and analytics

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pickle

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Import all sequencer components
try:
    from .adaptive_attack_sequencer import (
        AdaptiveAttackSequencer, AttackAction, AttackResult,
        AttackPhase, AttackVector, SequenceState, ResourceType
    )
    from .attack_sequence_planner import (
        MonteCarloTreeSearch, GeneticAlgorithmSequencer,
        DynamicProgrammingOptimizer, PlanningNode
    )
    from .rl_training import (
        DQNTrainer, ActorCriticTrainer, AttackEnvironmentSimulator,
        OnlineLearningSystem, TrainingConfig
    )
    SEQUENCER_COMPONENTS_AVAILABLE = True
except ImportError as e:
    SEQUENCER_COMPONENTS_AVAILABLE = False
    logging.warning(f"Sequencer components not available: {e}")

class AdaptiveAttackSequencerManager:
    """
    Main manager for the Adaptive Attack Sequencer system
    Coordinates all components and provides unified interface
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 models_dir: str = "./models",
                 enable_training: bool = True):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        if SEQUENCER_COMPONENTS_AVAILABLE:
            self._initialize_components(enable_training)
        else:
            self.logger.error("Sequencer components not available")
            self._initialize_fallback_components()
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_metrics = {}
        self.training_progress = {}
        
        self.logger.info("Adaptive Attack Sequencer Manager initialized")
    
    def _initialize_components(self, enable_training: bool):
        """Initialize all sequencer components"""
        # Main sequencer
        self.sequencer = AdaptiveAttackSequencer(
            state_size=self.config.get('state_size', 50),
            action_size=self.config.get('action_size', 100),
            learning_rate=self.config.get('learning_rate', 0.001)
        )
        
        # Planning algorithms
        self.mcts_planner = MonteCarloTreeSearch(
            exploration_weight=self.config.get('mcts_exploration_weight', 1.4)
        )
        
        self.genetic_planner = GeneticAlgorithmSequencer(
            population_size=self.config.get('ga_population_size', 50),
            mutation_rate=self.config.get('ga_mutation_rate', 0.1),
            crossover_rate=self.config.get('ga_crossover_rate', 0.8)
        )
        
        self.dp_optimizer = DynamicProgrammingOptimizer()
        
        # Training components (if enabled)
        if enable_training:
            training_config = TrainingConfig(
                batch_size=self.config.get('batch_size', 32),
                learning_rate=self.config.get('learning_rate', 0.001),
                gamma=self.config.get('gamma', 0.99)
            )
            
            self.dqn_trainer = DQNTrainer(
                state_size=self.config.get('state_size', 50),
                action_size=self.config.get('action_size', 100),
                config=training_config
            )
            
            self.ac_trainer = ActorCriticTrainer(
                state_size=self.config.get('state_size', 50),
                action_size=self.config.get('action_size', 100),
                config=training_config
            )
            
            self.online_learner = OnlineLearningSystem(self.dqn_trainer)
        else:
            self.dqn_trainer = None
            self.ac_trainer = None
            self.online_learner = None
        
        # Load pre-trained models if available
        self._load_pretrained_models()
    
    def _initialize_fallback_components(self):
        """Initialize fallback components when ML is not available"""
        self.sequencer = None
        self.mcts_planner = None
        self.genetic_planner = None
        self.dp_optimizer = None
        self.dqn_trainer = None
        self.ac_trainer = None
        self.online_learner = None
    
    async def optimize_attack_sequence(self,
                                     target_context: Dict[str, Any],
                                     available_attacks: List[AttackAction],
                                     optimization_method: str = "adaptive",
                                     constraints: Dict[str, Any] = None) -> List[AttackAction]:
        """
        Optimize attack sequence using specified method
        
        Args:
            target_context: Information about the target
            available_attacks: List of available attack actions
            optimization_method: Method to use ('adaptive', 'mcts', 'genetic', 'dp')
            constraints: Additional constraints for optimization
            
        Returns:
            Optimized attack sequence
        """
        self.logger.info(f"Optimizing sequence using {optimization_method} method")
        
        if not SEQUENCER_COMPONENTS_AVAILABLE:
            return self._fallback_optimization(available_attacks, target_context)
        
        try:
            if optimization_method == "adaptive" and self.sequencer:
                sequence = await self.sequencer.optimize_attack_sequence(
                    target_context, available_attacks, constraints
                )
            
            elif optimization_method == "mcts" and self.mcts_planner:
                # Create initial state for MCTS
                initial_state = SequenceState(
                    current_phase=AttackPhase.RECONNAISSANCE,
                    completed_actions=[],
                    pending_actions=available_attacks.copy(),
                    available_resources=self._get_default_resources(),
                    target_context=target_context
                )
                
                sequence = await self.mcts_planner.search(
                    initial_state, available_attacks,
                    max_iterations=self.config.get('mcts_iterations', 1000)
                )
            
            elif optimization_method == "genetic" and self.genetic_planner:
                sequence = await self.genetic_planner.evolve_sequence(
                    available_attacks, target_context,
                    generations=self.config.get('ga_generations', 100)
                )
            
            elif optimization_method == "dp" and self.dp_optimizer:
                time_budget = constraints.get('time_budget', 3600.0) if constraints else 3600.0
                sequence, _ = self.dp_optimizer.optimize_resource_allocation(
                    available_attacks, self._get_default_resources(), time_budget
                )
            
            else:
                self.logger.warning(f"Unknown or unavailable method {optimization_method}, using fallback")
                sequence = self._fallback_optimization(available_attacks, target_context)
            
            self.logger.info(f"Optimized sequence contains {len(sequence)} actions")
            return sequence
            
        except Exception as e:
            self.logger.error(f"Error in sequence optimization: {e}")
            return self._fallback_optimization(available_attacks, target_context)
    
    async def predict_action_success(self,
                                   action: AttackAction,
                                   context: Dict[str, Any]) -> float:
        """Predict success probability for an action"""
        if self.sequencer:
            return await self.sequencer.predict_success_probability(action, context)
        else:
            # Fallback prediction
            return self._fallback_success_prediction(action, context)
    
    async def adapt_sequence_realtime(self,
                                    current_sequence: List[AttackAction],
                                    current_state: SequenceState,
                                    feedback: Dict[str, Any]) -> List[AttackAction]:
        """Adapt sequence in real-time based on feedback"""
        if self.sequencer:
            return await self.sequencer.adjust_sequence_dynamically(
                current_sequence, current_state, feedback
            )
        else:
            # Simple fallback adaptation
            return self._fallback_adaptation(current_sequence, feedback)
    
    async def learn_from_execution(self,
                                 action: AttackAction,
                                 result: AttackResult,
                                 context: Dict[str, Any]):
        """Learn from action execution results"""
        if not SEQUENCER_COMPONENTS_AVAILABLE:
            return
        
        try:
            # Update main sequencer
            if self.sequencer:
                state_before = SequenceState(
                    current_phase=action.phase,
                    completed_actions=[],
                    pending_actions=[],
                    available_resources=self._get_default_resources(),
                    target_context=context
                )
                await self.sequencer.update_from_result(action, result, state_before)
            
            # Online learning
            if self.online_learner:
                await self.online_learner.adapt_online(action, result, context)
            
            # Track performance
            self._track_performance(action, result, context)
            
        except Exception as e:
            self.logger.error(f"Error in learning from execution: {e}")
    
    async def train_models(self,
                         training_data: List[Dict[str, Any]],
                         episodes: int = 1000) -> Dict[str, Any]:
        """Train RL models with provided data"""
        if not SEQUENCER_COMPONENTS_AVAILABLE or not self.dqn_trainer:
            return {"status": "Training not available"}
        
        self.logger.info(f"Starting training for {episodes} episodes")
        
        try:
            # Prepare training environment
            available_actions = self._extract_actions_from_data(training_data)
            target_contexts = self._extract_contexts_from_data(training_data)
            
            env_simulator = AttackEnvironmentSimulator(available_actions, target_contexts)
            
            # Train DQN
            dqn_results = await self._train_dqn(env_simulator, episodes)
            
            # Train Actor-Critic
            ac_results = await self._train_actor_critic(env_simulator, episodes // 2)
            
            # Save trained models
            self._save_models()
            
            training_results = {
                "status": "completed",
                "episodes": episodes,
                "dqn_performance": dqn_results,
                "actor_critic_performance": ac_results,
                "timestamp": time.time()
            }
            
            self.training_progress.update(training_results)
            self.logger.info("Training completed successfully")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in training: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        analytics = {
            "timestamp": time.time(),
            "component_status": {
                "sequencer_available": self.sequencer is not None,
                "mcts_available": self.mcts_planner is not None,
                "genetic_available": self.genetic_planner is not None,
                "dp_available": self.dp_optimizer is not None,
                "training_available": self.dqn_trainer is not None
            }
        }
        
        if self.sequencer:
            analytics["sequencer_analytics"] = self.sequencer.get_sequence_analytics()
        
        if self.performance_history:
            analytics["performance_summary"] = self._summarize_performance()
        
        if self.training_progress:
            analytics["training_status"] = self.training_progress
        
        analytics["adaptation_metrics"] = self.adaptation_metrics
        
        return analytics
    
    def export_trained_models(self, export_path: str) -> Dict[str, str]:
        """Export trained models for deployment"""
        export_dir = Path(export_path)
        export_dir.mkdir(exist_ok=True)
        
        exported_files = {}
        
        try:
            if self.dqn_trainer:
                dqn_path = export_dir / "dqn_model.pth"
                self.dqn_trainer.save_model(str(dqn_path))
                exported_files["dqn_model"] = str(dqn_path)
            
            if self.sequencer:
                sequencer_path = export_dir / "sequencer_state.pkl"
                with open(sequencer_path, 'wb') as f:
                    pickle.dump({
                        'attack_history': list(self.sequencer.attack_history),
                        'performance_metrics': self.performance_history,
                        'adaptation_metrics': self.adaptation_metrics
                    }, f)
                exported_files["sequencer_state"] = str(sequencer_path)
            
            # Export configuration
            config_path = export_dir / "sequencer_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            exported_files["config"] = str(config_path)
            
            self.logger.info(f"Models exported to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting models: {e}")
        
        return exported_files
    
    # Private helper methods
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "state_size": 50,
            "action_size": 100,
            "learning_rate": 0.001,
            "batch_size": 32,
            "gamma": 0.99,
            "mcts_exploration_weight": 1.4,
            "mcts_iterations": 1000,
            "ga_population_size": 50,
            "ga_mutation_rate": 0.1,
            "ga_crossover_rate": 0.8,
            "ga_generations": 100
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"Error loading config from {config_path}: {e}")
        
        return default_config
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        try:
            dqn_path = self.models_dir / "dqn_model.pth"
            if dqn_path.exists() and self.dqn_trainer:
                self.dqn_trainer.load_model(str(dqn_path))
                self.logger.info("Loaded pre-trained DQN model")
            
            sequencer_path = self.models_dir / "sequencer_state.pkl"
            if sequencer_path.exists() and self.sequencer:
                with open(sequencer_path, 'rb') as f:
                    state = pickle.load(f)
                    # Restore sequencer state
                    if 'attack_history' in state:
                        self.sequencer.attack_history.extend(state['attack_history'])
                self.logger.info("Loaded sequencer state")
                
        except Exception as e:
            self.logger.warning(f"Error loading pre-trained models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            if self.dqn_trainer:
                dqn_path = self.models_dir / "dqn_model.pth"
                self.dqn_trainer.save_model(str(dqn_path))
            
            if self.sequencer:
                sequencer_path = self.models_dir / "sequencer_state.pkl"
                with open(sequencer_path, 'wb') as f:
                    pickle.dump({
                        'attack_history': list(self.sequencer.attack_history),
                        'performance_metrics': self.performance_history,
                        'adaptation_metrics': self.adaptation_metrics
                    }, f)
                    
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def _get_default_resources(self) -> Dict[ResourceType, float]:
        """Get default resource allocation"""
        return {
            ResourceType.CPU_THREADS: 8.0,
            ResourceType.NETWORK_BANDWIDTH: 100.0,
            ResourceType.MEMORY: 1024.0,
            ResourceType.TIME_BUDGET: 3600.0,
            ResourceType.CONCURRENT_REQUESTS: 10.0
        }
    
    def _fallback_optimization(self, 
                             available_attacks: List[AttackAction],
                             target_context: Dict[str, Any]) -> List[AttackAction]:
        """Fallback optimization when ML components unavailable"""
        # Simple heuristic-based optimization
        scored_attacks = []
        
        for attack in available_attacks:
            score = (attack.priority * 0.4 + 
                    attack.success_probability * 0.4 +
                    (1.0 - attack.risk_score) * 0.2)
            scored_attacks.append((score, attack))
        
        scored_attacks.sort(key=lambda x: x[0], reverse=True)
        return [attack for _, attack in scored_attacks]
    
    def _fallback_success_prediction(self, 
                                   action: AttackAction,
                                   context: Dict[str, Any]) -> float:
        """Fallback success prediction"""
        base_prob = action.success_probability
        
        # Simple context adjustments
        if context.get('waf_detected'):
            base_prob *= 0.7
        
        return max(0.0, min(1.0, base_prob))
    
    def _fallback_adaptation(self, 
                           sequence: List[AttackAction],
                           feedback: Dict[str, Any]) -> List[AttackAction]:
        """Fallback sequence adaptation"""
        # Simple reordering based on feedback
        if feedback.get('major_failure'):
            # Shuffle sequence
            import random
            new_sequence = sequence.copy()
            random.shuffle(new_sequence)
            return new_sequence
        
        return sequence
    
    async def _train_dqn(self, env_simulator, episodes: int) -> Dict[str, Any]:
        """Train DQN model"""
        total_rewards = []
        
        for episode in range(episodes):
            initial_state = env_simulator.reset()
            total_reward, steps = await self.dqn_trainer.train_episode(
                env_simulator, initial_state
            )
            total_rewards.append(total_reward)
            
            if episode % 100 == 0:
                if NUMPY_AVAILABLE:
                    avg_reward = np.mean(total_rewards[-100:]) if total_rewards else 0
                else:
                    recent_rewards = total_rewards[-100:] if len(total_rewards) >= 100 else total_rewards
                    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                self.logger.debug(f"DQN Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        if NUMPY_AVAILABLE:
            avg_reward = np.mean(total_rewards) if total_rewards else 0
        else:
            avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
            
        return {
            "total_episodes": episodes,
            "average_reward": avg_reward,
            "final_epsilon": self.dqn_trainer.epsilon
        }
    
    async def _train_actor_critic(self, env_simulator, episodes: int) -> Dict[str, Any]:
        """Train Actor-Critic model"""
        total_rewards = []
        
        for episode in range(episodes):
            initial_state = env_simulator.reset()
            total_reward, steps = await self.ac_trainer.train_episode(
                env_simulator, initial_state
            )
            total_rewards.append(total_reward)
            
            if episode % 50 == 0:
                if NUMPY_AVAILABLE:
                    avg_reward = np.mean(total_rewards[-50:]) if total_rewards else 0
                else:
                    recent_rewards = total_rewards[-50:] if len(total_rewards) >= 50 else total_rewards
                    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                self.logger.debug(f"AC Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        if NUMPY_AVAILABLE:
            avg_reward = np.mean(total_rewards) if total_rewards else 0
        else:
            avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
            
        return {
            "total_episodes": episodes,
            "average_reward": avg_reward
        }
    
    def _extract_actions_from_data(self, training_data: List[Dict[str, Any]]) -> List[AttackAction]:
        """Extract attack actions from training data"""
        actions = []
        
        for data in training_data:
            if 'action' in data:
                action_data = data['action']
                action = AttackAction(
                    vector=AttackVector(action_data.get('vector', 'sql_injection')),
                    phase=AttackPhase(action_data.get('phase', 'exploitation')),
                    target_endpoint=action_data.get('target_endpoint', ''),
                    payload=action_data.get('payload', ''),
                    priority=action_data.get('priority', 0.5),
                    success_probability=action_data.get('success_probability', 0.5)
                )
                actions.append(action)
        
        return actions
    
    def _extract_contexts_from_data(self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract target contexts from training data"""
        contexts = []
        
        for data in training_data:
            if 'context' in data:
                contexts.append(data['context'])
        
        # Add default context if none found
        if not contexts:
            contexts.append({
                'technologies': ['php', 'mysql'],
                'waf_detected': False,
                'open_ports': [80, 443]
            })
        
        return contexts
    
    def _track_performance(self, 
                         action: AttackAction, 
                         result: AttackResult,
                         context: Dict[str, Any]):
        """Track performance metrics"""
        performance_entry = {
            'timestamp': time.time(),
            'action_vector': action.vector.value,
            'action_phase': action.phase.value,
            'success': result.success,
            'response_time': result.response_time,
            'confidence': result.confidence_score,
            'priority': action.priority,
            'context_size': len(context)
        }
        
        self.performance_history.append(performance_entry)
        
        # Keep only recent history
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-5000:]
    
    def _summarize_performance(self) -> Dict[str, Any]:
        """Summarize performance history"""
        if not self.performance_history:
            return {}
        
        if NUMPY_AVAILABLE:
            success_rate = np.mean([p['success'] for p in self.performance_history])
            avg_response_time = np.mean([p['response_time'] for p in self.performance_history])
            avg_confidence = np.mean([p['confidence'] for p in self.performance_history])
        else:
            successes = [p['success'] for p in self.performance_history]
            response_times = [p['response_time'] for p in self.performance_history]
            confidences = [p['confidence'] for p in self.performance_history]
            
            success_rate = sum(successes) / len(successes)
            avg_response_time = sum(response_times) / len(response_times)
            avg_confidence = sum(confidences) / len(confidences)
        
        # Performance by vector
        vector_performance = {}
        for entry in self.performance_history:
            vector = entry['action_vector']
            if vector not in vector_performance:
                vector_performance[vector] = []
            vector_performance[vector].append(entry['success'])
        
        if NUMPY_AVAILABLE:
            vector_success_rates = {
                vector: np.mean(successes)
                for vector, successes in vector_performance.items()
            }
        else:
            vector_success_rates = {
                vector: sum(successes) / len(successes)
                for vector, successes in vector_performance.items()
            }
        
        return {
            'total_actions': len(self.performance_history),
            'overall_success_rate': success_rate,
            'average_response_time': avg_response_time,
            'average_confidence': avg_confidence,
            'vector_success_rates': vector_success_rates
        }

# Create default instance
_default_manager = None

def get_sequencer_manager(**kwargs) -> AdaptiveAttackSequencerManager:
    """Get default sequencer manager instance"""
    global _default_manager
    if _default_manager is None:
        _default_manager = AdaptiveAttackSequencerManager(**kwargs)
    return _default_manager

# Export main classes
__all__ = [
    'AdaptiveAttackSequencerManager',
    'get_sequencer_manager',
    'SEQUENCER_COMPONENTS_AVAILABLE'
]