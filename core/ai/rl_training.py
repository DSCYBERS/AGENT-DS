#!/usr/bin/env python3
"""
Agent DS v2.0 - Reinforcement Learning Training
===============================================

Advanced reinforcement learning training system for attack sequencing:
- Deep Q-Network (DQN) training with experience replay
- Actor-Critic methods for continuous action spaces
- Policy Gradient methods for complex strategies
- Multi-Agent Reinforcement Learning for coordinated attacks
- Online learning and adaptation during penetration testing

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import json
import logging
import pickle
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Try to import RL dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Import sequencer components
from .adaptive_attack_sequencer import (
    AttackAction, AttackResult, AttackPhase, AttackVector,
    SequenceState, ResourceType, QNetwork
)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class TrainingConfig:
    """Configuration for RL training"""
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    memory_size: int = 10000
    hidden_size: int = 512
    num_episodes: int = 1000
    max_steps_per_episode: int = 50
    
class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy gradient methods"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state):
        # Shared layers
        x = F.relu(self.shared_fc1(state))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        
        # Actor network
        actor_x = F.relu(self.actor_fc(x))
        policy_logits = self.actor_out(actor_x)
        
        # Critic network
        critic_x = F.relu(self.critic_fc(x))
        value = self.critic_out(critic_x)
        
        return policy_logits, value

class DQNTrainer:
    """Deep Q-Network trainer for attack sequencing"""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 config: TrainingConfig):
        
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not RL_AVAILABLE:
            self.logger.error("RL dependencies not available")
            return
        
        # Initialize networks
        self.q_network = QNetwork(state_size, action_size, config.hidden_size)
        self.target_network = QNetwork(state_size, action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(config.memory_size)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.episode_rewards = []
        self.episode_losses = []
        self.training_step = 0
        
        # Update target network
        self.update_target_network()
        
        self.logger.info("DQN Trainer initialized")
    
    def update_target_network(self):
        """Update target network weights"""
        if RL_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if not RL_AVAILABLE:
            return random.randint(0, self.action_size - 1)
        
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, 
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def train_step(self) -> float:
        """Perform one training step"""
        if not RL_AVAILABLE or len(self.memory) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    async def train_episode(self, 
                           environment_simulator,
                           initial_state: np.ndarray) -> Tuple[float, int]:
        """Train for one episode"""
        state = initial_state
        total_reward = 0.0
        steps = 0
        
        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.select_action(state, training=True)
            
            # Execute action in environment
            next_state, reward, done = await environment_simulator.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        return total_reward, steps
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if RL_AVAILABLE:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'episode_rewards': self.episode_rewards,
                'config': self.config
            }, filepath)
            self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if RL_AVAILABLE:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            self.episode_rewards = checkpoint['episode_rewards']
            self.logger.info(f"Model loaded from {filepath}")

class ActorCriticTrainer:
    """Actor-Critic trainer for policy gradient methods"""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 config: TrainingConfig):
        
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not RL_AVAILABLE:
            self.logger.error("RL dependencies not available")
            return
        
        # Initialize network
        self.network = ActorCriticNetwork(state_size, action_size, config.hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # Training state
        self.episode_rewards = []
        self.episode_losses = []
        
        self.logger.info("Actor-Critic Trainer initialized")
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using policy network"""
        if not RL_AVAILABLE:
            return random.randint(0, self.action_size - 1), 0.0, 0.0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            
        # Sample action from policy
        policy = F.softmax(policy_logits, dim=-1)
        action_dist = Categorical(policy)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action).item(), value.item()
    
    async def train_episode(self,
                           environment_simulator,
                           initial_state: np.ndarray) -> Tuple[float, int]:
        """Train for one episode using Actor-Critic"""
        if not RL_AVAILABLE:
            return 0.0, 0
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        state = initial_state
        total_reward = 0.0
        
        # Collect episode data
        for step in range(self.config.max_steps_per_episode):
            action, log_prob, value = self.select_action(state)
            
            next_state, reward, done = await environment_simulator.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards)
        advantages = self._calculate_advantages(returns, values)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        log_probs_tensor = torch.FloatTensor(log_probs)
        
        # Forward pass
        policy_logits, values_pred = self.network(states_tensor)
        
        # Calculate losses
        policy_loss = self._calculate_policy_loss(
            policy_logits, actions_tensor, advantages_tensor
        )
        value_loss = F.mse_loss(values_pred.squeeze(), returns_tensor)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_reward, len(states)
    
    def _calculate_returns(self, rewards: List[float]) -> List[float]:
        """Calculate discounted returns"""
        returns = []
        G = 0
        
        for reward in reversed(rewards):
            G = reward + self.config.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def _calculate_advantages(self, 
                             returns: List[float], 
                             values: List[float]) -> List[float]:
        """Calculate advantages (returns - values)"""
        advantages = []
        for ret, val in zip(returns, values):
            advantages.append(ret - val)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.tolist()
    
    def _calculate_policy_loss(self,
                              policy_logits: torch.Tensor,
                              actions: torch.Tensor,
                              advantages: torch.Tensor) -> torch.Tensor:
        """Calculate policy loss"""
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        policy_loss = -(action_log_probs * advantages).mean()
        
        return policy_loss

class AttackEnvironmentSimulator:
    """Simulates attack environment for RL training"""
    
    def __init__(self, 
                 available_actions: List[AttackAction],
                 target_contexts: List[Dict[str, Any]]):
        
        self.available_actions = available_actions
        self.target_contexts = target_contexts
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Current episode state
        self.current_target_context = None
        self.current_sequence_state = None
        self.episode_step = 0
        self.max_episode_steps = 20
        
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        # Select random target context
        self.current_target_context = random.choice(self.target_contexts)
        
        # Initialize sequence state
        self.current_sequence_state = SequenceState(
            current_phase=AttackPhase.RECONNAISSANCE,
            completed_actions=[],
            pending_actions=self.available_actions.copy(),
            available_resources={
                ResourceType.CPU_THREADS: 8.0,
                ResourceType.NETWORK_BANDWIDTH: 100.0,
                ResourceType.MEMORY: 1024.0,
                ResourceType.TIME_BUDGET: 3600.0,
                ResourceType.CONCURRENT_REQUESTS: 10.0
            },
            target_context=self.current_target_context
        )
        
        self.episode_step = 0
        
        return self._encode_state()
    
    async def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return next state, reward, done"""
        if action_idx >= len(self.available_actions):
            # Invalid action
            return self._encode_state(), -1.0, True
        
        action = self.available_actions[action_idx]
        
        # Simulate action execution
        result = await self._simulate_action_execution(action)
        
        # Update state
        self.current_sequence_state.completed_actions.append(result)
        if action in self.current_sequence_state.pending_actions:
            self.current_sequence_state.pending_actions.remove(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, result)
        
        # Check if episode is done
        self.episode_step += 1
        done = (self.episode_step >= self.max_episode_steps or 
                not self.current_sequence_state.pending_actions or
                self._is_mission_complete())
        
        return self._encode_state(), reward, done
    
    async def _simulate_action_execution(self, action: AttackAction) -> AttackResult:
        """Simulate execution of an attack action"""
        # Base success probability
        success_prob = action.success_probability
        
        # Context-based adjustments
        if self.current_target_context.get('waf_detected'):
            success_prob *= 0.7
        
        if action.phase != self.current_sequence_state.current_phase:
            # Phase transition penalty
            success_prob *= 0.8
        
        # Simulate execution
        success = random.random() < success_prob
        response_time = random.uniform(0.1, 3.0)
        
        return AttackResult(
            action=action,
            success=success,
            response_time=response_time,
            status_code=200 if success else random.choice([403, 404, 500]),
            response_size=random.randint(100, 5000),
            confidence_score=success_prob if success else 1.0 - success_prob
        )
    
    def _encode_state(self) -> np.ndarray:
        """Encode current state as feature vector"""
        state = np.zeros(50)  # Fixed size state vector
        
        # Target context features
        state[0] = len(self.current_target_context.get('technologies', []))
        state[1] = 1.0 if self.current_target_context.get('waf_detected') else 0.0
        state[2] = len(self.current_target_context.get('open_ports', []))
        
        # Sequence state features
        state[3] = len(self.current_sequence_state.completed_actions)
        state[4] = len(self.current_sequence_state.pending_actions)
        state[5] = self.current_sequence_state.success_rate
        state[6] = self.episode_step / self.max_episode_steps
        
        # Phase encoding
        phase_idx = list(AttackPhase).index(self.current_sequence_state.current_phase)
        state[7 + phase_idx] = 1.0  # One-hot encoding
        
        # Resource state
        for i, (resource_type, amount) in enumerate(self.current_sequence_state.available_resources.items()):
            if i < 5:  # Limit to first 5 resources
                state[16 + i] = amount / 1000.0  # Normalize
        
        return state
    
    def _calculate_reward(self, action: AttackAction, result: AttackResult) -> float:
        """Calculate reward for action execution"""
        reward = 0.0
        
        # Success reward
        if result.success:
            reward += 10.0
            
            # Bonus for high-priority actions
            reward += action.priority * 5.0
            
            # Phase progression bonus
            if action.phase == self.current_sequence_state.current_phase:
                reward += 2.0
        else:
            reward -= 1.0
        
        # Time efficiency reward
        if result.response_time < 1.0:
            reward += 1.0
        elif result.response_time > 3.0:
            reward -= 1.0
        
        # Risk penalty
        reward -= action.risk_score * 2.0
        
        return reward
    
    def _is_mission_complete(self) -> bool:
        """Check if mission objectives are complete"""
        # Simple completion check based on phase progression
        completed_phases = set(
            result.action.phase 
            for result in self.current_sequence_state.completed_actions
            if result.success
        )
        
        # Consider mission complete if we've successfully completed 3+ phases
        return len(completed_phases) >= 3

class OnlineLearningSystem:
    """Online learning system for real-time adaptation"""
    
    def __init__(self, base_trainer):
        self.base_trainer = base_trainer
        self.online_buffer = deque(maxlen=1000)
        self.adaptation_rate = 0.01
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def adapt_online(self, 
                          action: AttackAction,
                          result: AttackResult,
                          context: Dict[str, Any]):
        """Adapt model based on real-time feedback"""
        if not RL_AVAILABLE:
            return
        
        # Encode experience
        state = self._encode_context(context)
        action_idx = self._action_to_index(action)
        reward = self._calculate_adaptation_reward(result)
        
        # Store in online buffer
        self.online_buffer.append((state, action_idx, reward))
        
        # Perform mini-batch online update
        if len(self.online_buffer) >= 32:
            await self._online_update()
    
    async def _online_update(self):
        """Perform online model update"""
        if not hasattr(self.base_trainer, 'q_network'):
            return
        
        # Sample from online buffer
        batch = random.sample(self.online_buffer, min(16, len(self.online_buffer)))
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        
        # Quick update with small learning rate
        optimizer = optim.Adam(self.base_trainer.q_network.parameters(), 
                             lr=self.adaptation_rate)
        
        current_q_values = self.base_trainer.q_network(states).gather(1, actions.unsqueeze(1))
        target_q_values = rewards.unsqueeze(1)
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.logger.debug(f"Online adaptation update, loss: {loss.item():.4f}")
    
    def _encode_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context for online learning"""
        # Simplified encoding for online learning
        state = np.zeros(20)
        
        state[0] = len(context.get('technologies', []))
        state[1] = 1.0 if context.get('waf_detected') else 0.0
        state[2] = context.get('response_time_avg', 0.0) / 1000.0
        
        return state
    
    def _action_to_index(self, action: AttackAction) -> int:
        """Convert action to index"""
        # Simple mapping based on attack vector
        vector_to_idx = {v: i for i, v in enumerate(AttackVector)}
        return vector_to_idx.get(action.vector, 0)
    
    def _calculate_adaptation_reward(self, result: AttackResult) -> float:
        """Calculate reward for adaptation"""
        base_reward = 1.0 if result.success else -0.5
        confidence_bonus = result.confidence_score * 0.5
        return base_reward + confidence_bonus

# Export main classes
__all__ = [
    'DQNTrainer', 
    'ActorCriticTrainer', 
    'AttackEnvironmentSimulator',
    'OnlineLearningSystem',
    'TrainingConfig',
    'ReplayBuffer'
]