#!/usr/bin/env python3
"""
Agent DS v2.0 - Adaptive Attack Sequencer
==========================================

Advanced reinforcement learning system for optimal attack ordering with:
- Success probability prediction using ensemble models
- Dynamic phase adjustment based on real-time feedback
- Intelligent resource allocation for maximum effectiveness
- Q-learning based attack sequence optimization
- Multi-armed bandit for exploit selection
- Contextual bandits for target-specific strategies

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import json
import logging
import numpy as np
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque

# Try to import AI dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class AttackPhase(Enum):
    """Attack phases in penetration testing"""
    RECONNAISSANCE = "reconnaissance"
    ENUMERATION = "enumeration"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    DATA_EXFILTRATION = "data_exfiltration"
    CLEANUP = "cleanup"

class AttackVector(Enum):
    """Available attack vectors"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    SSRF = "ssrf"
    DIRECTORY_TRAVERSAL = "directory_traversal"
    COMMAND_INJECTION = "command_injection"
    FILE_UPLOAD = "file_upload"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    BRUTE_FORCE = "brute_force"
    SOCIAL_ENGINEERING = "social_engineering"

class ResourceType(Enum):
    """System resources for attack allocation"""
    CPU_THREADS = "cpu_threads"
    NETWORK_BANDWIDTH = "network_bandwidth"
    MEMORY = "memory"
    TIME_BUDGET = "time_budget"
    CONCURRENT_REQUESTS = "concurrent_requests"

@dataclass
class AttackAction:
    """Represents an individual attack action"""
    vector: AttackVector
    phase: AttackPhase
    target_endpoint: str
    payload: str
    priority: float = 0.0
    estimated_time: float = 0.0
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    success_probability: float = 0.0
    risk_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttackResult:
    """Result of an attack action execution"""
    action: AttackAction
    success: bool
    response_time: float
    status_code: int
    response_size: int
    error_message: Optional[str] = None
    indicators_found: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SequenceState:
    """Current state of attack sequence"""
    current_phase: AttackPhase
    completed_actions: List[AttackResult]
    pending_actions: List[AttackAction]
    available_resources: Dict[ResourceType, float]
    target_context: Dict[str, Any]
    success_rate: float = 0.0
    total_time_elapsed: float = 0.0
    phase_completion: Dict[AttackPhase, float] = field(default_factory=dict)

class QNetwork(nn.Module):
    """Deep Q-Network for attack sequence optimization"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MultiArmedBandit:
    """Multi-armed bandit for exploit selection"""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0
        
    def select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy"""
        if random.random() < self.epsilon or self.total_count == 0:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.values)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics"""
        self.counts[arm] += 1
        self.total_count += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
    
    def get_best_arm(self) -> int:
        """Get the arm with highest average reward"""
        return np.argmax(self.values)

class ContextualBandit:
    """Contextual bandit for target-specific strategies"""
    
    def __init__(self, n_features: int, n_arms: int, alpha: float = 1.0):
        self.n_features = n_features
        self.n_arms = n_arms
        self.alpha = alpha
        
        # Initialize parameters for each arm
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
        
    def select_arm(self, context: np.ndarray) -> int:
        """Select arm based on context using LinUCB algorithm"""
        ucb_values = []
        
        for arm in range(self.n_arms):
            # Compute confidence bound
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            
            # Upper confidence bound
            cb = self.alpha * np.sqrt(context.T @ A_inv @ context)
            ucb = theta.T @ context + cb
            ucb_values.append(ucb)
            
        return np.argmax(ucb_values)
    
    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update contextual bandit parameters"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

class AdaptiveAttackSequencer:
    """
    Advanced reinforcement learning system for optimal attack ordering
    """
    
    def __init__(self, 
                 state_size: int = 50,
                 action_size: int = 100,
                 learning_rate: float = 0.001,
                 epsilon: float = 0.1,
                 gamma: float = 0.95):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Initialize models if ML is available
        if ML_AVAILABLE:
            self.q_network = QNetwork(state_size, action_size)
            self.target_network = QNetwork(state_size, action_size)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            
            # Success prediction models
            self.success_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.time_predictor = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.risk_predictor = LogisticRegression(random_state=42)
            self.scaler = StandardScaler()
            
            # Initialize bandits
            self.exploit_bandit = MultiArmedBandit(len(AttackVector), epsilon=0.1)
            self.phase_bandit = MultiArmedBandit(len(AttackPhase), epsilon=0.1)
            self.contextual_bandit = ContextualBandit(n_features=20, n_arms=len(AttackVector))
            
        else:
            self.logger.warning("ML dependencies not available, using heuristic methods")
            self.q_network = None
            
        # Attack sequence management
        self.attack_history = deque(maxlen=10000)
        self.sequence_memory = []
        self.current_state = None
        self.phase_transitions = defaultdict(list)
        self.success_patterns = defaultdict(list)
        
        # Resource management
        self.resource_allocator = ResourceAllocator()
        self.performance_monitor = PerformanceMonitor()
        
        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.exploration_decay = 0.995
        self.update_frequency = 10
        self.model_retrain_threshold = 100
        
        self.logger.info("Adaptive Attack Sequencer initialized")
    
    async def optimize_attack_sequence(self, 
                                     target_context: Dict[str, Any],
                                     available_attacks: List[AttackAction],
                                     constraints: Dict[str, Any] = None) -> List[AttackAction]:
        """
        Optimize attack sequence using reinforcement learning
        """
        self.logger.info(f"Optimizing sequence for {len(available_attacks)} attacks")
        
        # Initialize state
        state = self._encode_state(target_context, available_attacks)
        
        if ML_AVAILABLE and self.q_network:
            # Use deep Q-learning for sequence optimization
            optimized_sequence = await self._deep_q_optimization(
                state, available_attacks, constraints
            )
        else:
            # Use heuristic optimization
            optimized_sequence = await self._heuristic_optimization(
                available_attacks, target_context, constraints
            )
        
        # Apply contextual bandit recommendations
        if ML_AVAILABLE:
            optimized_sequence = self._apply_contextual_recommendations(
                optimized_sequence, target_context
            )
        
        self.logger.info(f"Optimized sequence with {len(optimized_sequence)} actions")
        return optimized_sequence
    
    async def predict_success_probability(self, 
                                        action: AttackAction,
                                        context: Dict[str, Any]) -> float:
        """
        Predict success probability for an attack action
        """
        if not ML_AVAILABLE or not hasattr(self.success_predictor, 'predict_proba'):
            # Use heuristic prediction
            return self._heuristic_success_prediction(action, context)
        
        try:
            # Encode features
            features = self._encode_action_features(action, context)
            features_scaled = self.scaler.transform([features])
            
            # Predict probability
            prob = self.success_predictor.predict_proba(features_scaled)[0][1]
            
            # Apply contextual adjustments
            contextual_factor = self._calculate_contextual_factor(action, context)
            adjusted_prob = prob * contextual_factor
            
            return max(0.0, min(1.0, adjusted_prob))
            
        except Exception as e:
            self.logger.error(f"Error predicting success probability: {e}")
            return self._heuristic_success_prediction(action, context)
    
    async def update_from_result(self, 
                               action: AttackAction, 
                               result: AttackResult,
                               state_before: SequenceState):
        """
        Update models based on attack result
        """
        self.logger.debug(f"Updating from result: {result.success}")
        
        # Store in history
        self.attack_history.append({
            'action': action,
            'result': result,
            'state': state_before,
            'timestamp': time.time()
        })
        
        if ML_AVAILABLE:
            # Update bandits
            reward = self._calculate_reward(result)
            
            # Update exploit bandit
            arm_idx = list(AttackVector).index(action.vector)
            self.exploit_bandit.update(arm_idx, reward)
            
            # Update phase bandit
            phase_idx = list(AttackPhase).index(action.phase)
            self.phase_bandit.update(phase_idx, reward)
            
            # Update contextual bandit
            context_vector = self._encode_context_vector(state_before.target_context)
            self.contextual_bandit.update(arm_idx, context_vector, reward)
            
            # Update success patterns
            self._update_success_patterns(action, result, state_before)
            
            # Retrain models if threshold reached
            if len(self.attack_history) % self.model_retrain_threshold == 0:
                await self._retrain_models()
        
        # Update performance metrics
        self.performance_monitor.update(action, result)
        
        # Decay exploration
        self.epsilon *= self.exploration_decay
    
    async def adjust_sequence_dynamically(self, 
                                        current_sequence: List[AttackAction],
                                        current_state: SequenceState,
                                        real_time_feedback: Dict[str, Any]) -> List[AttackAction]:
        """
        Dynamically adjust attack sequence based on real-time feedback
        """
        self.logger.info("Adjusting sequence based on real-time feedback")
        
        # Analyze feedback
        feedback_analysis = self._analyze_feedback(real_time_feedback)
        
        # Check if major adjustment needed
        if feedback_analysis.get('major_adjustment_needed', False):
            # Re-optimize entire sequence
            return await self.optimize_attack_sequence(
                current_state.target_context,
                current_sequence,
                constraints=feedback_analysis.get('new_constraints')
            )
        
        # Make incremental adjustments
        adjusted_sequence = self._make_incremental_adjustments(
            current_sequence, feedback_analysis, current_state
        )
        
        self.logger.info(f"Applied {len(feedback_analysis.get('adjustments', []))} adjustments")
        return adjusted_sequence
    
    def allocate_resources_optimally(self, 
                                   actions: List[AttackAction],
                                   available_resources: Dict[ResourceType, float]) -> Dict[str, Dict[ResourceType, float]]:
        """
        Optimally allocate resources across attack actions
        """
        return self.resource_allocator.allocate(actions, available_resources)
    
    def get_sequence_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about attack sequence performance
        """
        if not self.attack_history:
            return {"status": "No data available"}
        
        total_attacks = len(self.attack_history)
        successful_attacks = sum(1 for h in self.attack_history if h['result'].success)
        
        analytics = {
            "total_attacks": total_attacks,
            "success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0,
            "average_response_time": np.mean([h['result'].response_time for h in self.attack_history]),
            "phase_performance": self._analyze_phase_performance(),
            "vector_performance": self._analyze_vector_performance(),
            "learning_progress": self._analyze_learning_progress(),
            "resource_efficiency": self.resource_allocator.get_efficiency_metrics(),
            "adaptation_metrics": self._get_adaptation_metrics()
        }
        
        if ML_AVAILABLE:
            analytics.update({
                "model_confidence": self._get_model_confidence(),
                "bandit_performance": self._get_bandit_performance(),
                "prediction_accuracy": self._get_prediction_accuracy()
            })
        
        return analytics
    
    # Private helper methods
    
    def _encode_state(self, 
                     target_context: Dict[str, Any], 
                     available_attacks: List[AttackAction]) -> np.ndarray:
        """Encode current state for ML models"""
        state = np.zeros(self.state_size)
        
        # Encode target context
        state[0] = len(target_context.get('technologies', []))
        state[1] = 1.0 if target_context.get('waf_detected') else 0.0
        state[2] = len(target_context.get('open_ports', []))
        state[3] = target_context.get('response_time_avg', 0.0) / 1000.0  # Normalize
        
        # Encode available attacks
        for i, attack in enumerate(available_attacks[:10]):  # Limit to first 10
            base_idx = 4 + i * 4
            if base_idx + 3 < self.state_size:
                state[base_idx] = attack.priority
                state[base_idx + 1] = attack.success_probability
                state[base_idx + 2] = attack.estimated_time / 60.0  # Normalize to minutes
                state[base_idx + 3] = attack.risk_score
        
        return state
    
    async def _deep_q_optimization(self, 
                                 state: np.ndarray,
                                 available_attacks: List[AttackAction],
                                 constraints: Dict[str, Any]) -> List[AttackAction]:
        """Use deep Q-learning for sequence optimization"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Get top actions based on Q-values
        top_actions_idx = torch.argsort(q_values, descending=True).squeeze()
        
        # Map back to actual attacks
        optimized_sequence = []
        for idx in top_actions_idx[:len(available_attacks)]:
            if idx < len(available_attacks):
                optimized_sequence.append(available_attacks[idx])
        
        return optimized_sequence
    
    async def _heuristic_optimization(self, 
                                    available_attacks: List[AttackAction],
                                    target_context: Dict[str, Any],
                                    constraints: Dict[str, Any]) -> List[AttackAction]:
        """Heuristic-based sequence optimization"""
        # Score each attack
        scored_attacks = []
        for attack in available_attacks:
            score = self._calculate_heuristic_score(attack, target_context)
            scored_attacks.append((score, attack))
        
        # Sort by score
        scored_attacks.sort(key=lambda x: x[0], reverse=True)
        
        return [attack for _, attack in scored_attacks]
    
    def _calculate_heuristic_score(self, 
                                  action: AttackAction,
                                  context: Dict[str, Any]) -> float:
        """Calculate heuristic score for an attack action"""
        score = 0.0
        
        # Base priority
        score += action.priority * 0.3
        
        # Success probability
        score += action.success_probability * 0.4
        
        # Time efficiency (inverse of time)
        if action.estimated_time > 0:
            score += (1.0 / action.estimated_time) * 0.2
        
        # Risk consideration (lower risk = higher score)
        score += (1.0 - action.risk_score) * 0.1
        
        return score
    
    def _heuristic_success_prediction(self, 
                                    action: AttackAction,
                                    context: Dict[str, Any]) -> float:
        """Heuristic success probability prediction"""
        base_prob = 0.5  # Default probability
        
        # Adjust based on attack vector
        vector_adjustments = {
            AttackVector.SQL_INJECTION: 0.7,
            AttackVector.XSS: 0.6,
            AttackVector.BRUTE_FORCE: 0.3,
            AttackVector.SSRF: 0.5
        }
        base_prob = vector_adjustments.get(action.vector, base_prob)
        
        # Context adjustments
        if context.get('waf_detected'):
            base_prob *= 0.6
        
        if context.get('technologies'):
            if 'wordpress' in str(context.get('technologies')).lower():
                base_prob *= 1.2  # WordPress often has more vulnerabilities
        
        return max(0.0, min(1.0, base_prob))

# Additional helper classes

class ResourceAllocator:
    """Manages optimal resource allocation"""
    
    def __init__(self):
        self.allocation_history = []
        self.efficiency_metrics = defaultdict(list)
    
    def allocate(self, 
                actions: List[AttackAction],
                available_resources: Dict[ResourceType, float]) -> Dict[str, Dict[ResourceType, float]]:
        """Allocate resources optimally across actions"""
        
        total_requirements = defaultdict(float)
        for action in actions:
            for resource_type, amount in action.resource_requirements.items():
                total_requirements[resource_type] += amount
        
        allocations = {}
        for action in actions:
            action_id = f"{action.vector.value}_{hash(action.target_endpoint)}"
            allocations[action_id] = {}
            
            for resource_type, required in action.resource_requirements.items():
                available = available_resources.get(resource_type, 0.0)
                total_req = total_requirements[resource_type]
                
                if total_req > 0:
                    # Proportional allocation
                    ratio = min(1.0, available / total_req)
                    allocated = required * ratio
                else:
                    allocated = 0.0
                
                allocations[action_id][resource_type] = allocated
        
        return allocations
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get resource allocation efficiency metrics"""
        if not self.efficiency_metrics:
            return {}
        
        return {
            "average_utilization": np.mean([
                np.mean(list(alloc.values())) 
                for alloc in self.allocation_history
            ]) if self.allocation_history else 0.0,
            "allocation_count": len(self.allocation_history)
        }

class PerformanceMonitor:
    """Monitors attack performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.phase_metrics = defaultdict(lambda: defaultdict(list))
    
    def update(self, action: AttackAction, result: AttackResult):
        """Update performance metrics"""
        self.metrics['success_rate'].append(1.0 if result.success else 0.0)
        self.metrics['response_time'].append(result.response_time)
        self.metrics['confidence'].append(result.confidence_score)
        
        # Phase-specific metrics
        phase = action.phase.value
        self.phase_metrics[phase]['success_rate'].append(1.0 if result.success else 0.0)
        self.phase_metrics[phase]['response_time'].append(result.response_time)

# Export main class
__all__ = ['AdaptiveAttackSequencer', 'AttackAction', 'AttackResult', 'SequenceState', 
           'AttackPhase', 'AttackVector', 'ResourceType']