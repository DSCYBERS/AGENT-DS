"""
Agent DS - Reinforcement Learning Attack Engine
Advanced AI-driven attack sequencing with reward/penalty learning system
"""

import asyncio
import json
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import random
from collections import deque, defaultdict

# ML/RL imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    import numpy as np
except ImportError:
    torch = None
    nn = None

from core.ai_learning.autonomous_engine import AutonomousLearningEngine
from core.config.settings import Config
from core.utils.logger import get_logger
from core.database.manager import DatabaseManager

logger = get_logger('rl_engine')

@dataclass
class AttackState:
    """Represents the current state of an attack sequence"""
    target_info: Dict[str, Any]
    discovered_services: List[str]
    identified_vulns: List[str]
    attempted_attacks: List[str]
    success_history: List[bool]
    current_phase: str
    time_elapsed: float
    detection_risk: float
    available_tools: List[str]
    target_responses: List[Dict]

@dataclass
class AttackAction:
    """Represents an attack action that can be taken"""
    action_id: str
    attack_type: str
    tool_name: str
    payload: str
    target_endpoint: str
    evasion_techniques: List[str]
    expected_reward: float
    risk_level: float

@dataclass
class AttackReward:
    """Reward structure for RL training"""
    base_reward: float
    success_bonus: float
    stealth_bonus: float
    efficiency_bonus: float
    innovation_bonus: float
    detection_penalty: float
    time_penalty: float
    total_reward: float

class DQNAgent(nn.Module):
    """Deep Q-Network for attack sequence optimization"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DQNAgent, self).__init__()
        
        if not torch:
            raise ImportError("PyTorch is required for RL functionality")
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReinforcementLearningEngine:
    """Main RL engine for Agent DS attack optimization"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        self.base_learning_engine = AutonomousLearningEngine()
        self.logger = get_logger('rl_attack_engine')
        
        # RL configuration
        self.rl_config = self.config.get('reinforcement_learning', {})
        self.state_size = self.rl_config.get('state_size', 128)
        self.action_size = self.rl_config.get('action_size', 100)
        self.learning_rate = self.rl_config.get('learning_rate', 0.001)
        self.epsilon = self.rl_config.get('epsilon_start', 1.0)
        self.epsilon_decay = self.rl_config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.rl_config.get('epsilon_min', 0.01)
        self.batch_size = self.rl_config.get('batch_size', 32)
        self.memory_size = self.rl_config.get('memory_size', 10000)
        
        # Initialize DQN if torch is available
        if torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DQNAgent(self.state_size, self.action_size).to(self.device)
            self.target_network = DQNAgent(self.state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Attack action space
        self.action_space = self._initialize_action_space()
        
        # Reward system configuration
        self.reward_config = {
            'successful_exploit': 100.0,
            'partial_success': 50.0,
            'information_disclosure': 25.0,
            'failed_attack': -10.0,
            'detection_trigger': -50.0,
            'time_efficiency_bonus': 20.0,
            'stealth_bonus': 30.0,
            'innovation_bonus': 40.0,
            'chained_exploit_bonus': 75.0
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.success_rates = []
        self.training_metrics = defaultdict(list)
    
    def _initialize_action_space(self) -> List[AttackAction]:
        """Initialize the space of possible attack actions"""
        actions = []
        
        # SQL Injection actions
        sql_payloads = [
            "' UNION SELECT 1,2,3--",
            "' OR '1'='1'--",
            "'; DROP TABLE users;--",
            "' UNION SELECT user(),database(),version()--",
            "' AND (SELECT SUBSTRING(@@version,1,1))='5'--"
        ]
        
        for i, payload in enumerate(sql_payloads):
            actions.append(AttackAction(
                action_id=f"sql_{i}",
                attack_type="sql_injection",
                tool_name="sqlmap",
                payload=payload,
                target_endpoint="auto",
                evasion_techniques=["encoding", "case_variation"],
                expected_reward=0.0,
                risk_level=0.3
            ))
        
        # XSS actions
        xss_payloads = [
            '<script>alert("XSS")</script>',
            '<img src=x onerror=alert("XSS")>',
            'javascript:alert("XSS")',
            '<svg onload=alert("XSS")>',
            '"><script>alert("XSS")</script>'
        ]
        
        for i, payload in enumerate(xss_payloads):
            actions.append(AttackAction(
                action_id=f"xss_{i}",
                attack_type="xss",
                tool_name="custom",
                payload=payload,
                target_endpoint="auto",
                evasion_techniques=["encoding", "obfuscation"],
                expected_reward=0.0,
                risk_level=0.2
            ))
        
        # Command injection actions
        cmd_payloads = [
            "; cat /etc/passwd",
            "| whoami",
            "`id`",
            "; ls -la",
            "&& ping -c 1 attacker.com"
        ]
        
        for i, payload in enumerate(cmd_payloads):
            actions.append(AttackAction(
                action_id=f"cmd_{i}",
                attack_type="command_injection",
                tool_name="custom",
                payload=payload,
                target_endpoint="auto",
                evasion_techniques=["encoding", "timing"],
                expected_reward=0.0,
                risk_level=0.4
            ))
        
        # File inclusion actions
        lfi_payloads = [
            "../../../etc/passwd",
            "....//....//....//etc/passwd",
            "/etc/passwd%00",
            "php://filter/read=convert.base64-encode/resource=index.php",
            "data://text/plain;base64,PD9waHAgcGhwaW5mbygpOyA/Pg=="
        ]
        
        for i, payload in enumerate(lfi_payloads):
            actions.append(AttackAction(
                action_id=f"lfi_{i}",
                attack_type="file_inclusion",
                tool_name="custom",
                payload=payload,
                target_endpoint="auto",
                evasion_techniques=["encoding", "null_bytes"],
                expected_reward=0.0,
                risk_level=0.3
            ))
        
        # SSRF actions
        ssrf_payloads = [
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost:22",
            "http://127.0.0.1:3306",
            "file:///etc/passwd",
            "gopher://127.0.0.1:25/_HELO%20example.com"
        ]
        
        for i, payload in enumerate(ssrf_payloads):
            actions.append(AttackAction(
                action_id=f"ssrf_{i}",
                attack_type="ssrf",
                tool_name="custom",
                payload=payload,
                target_endpoint="auto",
                evasion_techniques=["ip_encoding", "dns_rebinding"],
                expected_reward=0.0,
                risk_level=0.4
            ))
        
        return actions
    
    def encode_state(self, attack_state: AttackState) -> np.ndarray:
        """Encode attack state into numerical vector for neural network"""
        try:
            state_vector = np.zeros(self.state_size)
            
            # Encode target information (0-20)
            state_vector[0] = len(attack_state.discovered_services)
            state_vector[1] = len(attack_state.identified_vulns)
            state_vector[2] = len(attack_state.attempted_attacks)
            state_vector[3] = sum(attack_state.success_history) if attack_state.success_history else 0
            state_vector[4] = len(attack_state.success_history) if attack_state.success_history else 0
            state_vector[5] = attack_state.time_elapsed / 3600.0  # Normalize to hours
            state_vector[6] = attack_state.detection_risk
            state_vector[7] = len(attack_state.available_tools)
            
            # Encode service types (8-30)
            common_services = [
                'http', 'https', 'ssh', 'ftp', 'mysql', 'postgresql', 'mongodb',
                'redis', 'apache', 'nginx', 'iis', 'php', 'python', 'java',
                'nodejs', 'wordpress', 'drupal', 'joomla', 'tomcat', 'jenkins',
                'docker', 'kubernetes'
            ]
            
            for i, service in enumerate(common_services[:22]):
                if any(service in svc.lower() for svc in attack_state.discovered_services):
                    state_vector[8 + i] = 1.0
            
            # Encode vulnerability types (30-50)
            vuln_types = [
                'sql_injection', 'xss', 'rce', 'lfi', 'ssrf', 'xxe', 'ssti',
                'deserialization', 'csrf', 'idor', 'auth_bypass', 'privilege_escalation',
                'directory_traversal', 'file_upload', 'weak_auth', 'info_disclosure',
                'cors', 'clickjacking', 'open_redirect', 'cache_poisoning'
            ]
            
            for i, vuln_type in enumerate(vuln_types):
                if any(vuln_type in vuln.lower() for vuln in attack_state.identified_vulns):
                    state_vector[30 + i] = 1.0
            
            # Encode attack history (50-80)
            attack_types = [
                'sql_injection', 'xss', 'command_injection', 'file_inclusion',
                'ssrf', 'xxe', 'ssti', 'deserialization', 'csrf', 'auth_bypass',
                'brute_force', 'directory_enum', 'subdomain_enum', 'port_scan',
                'web_scan', 'vuln_scan', 'exploit', 'post_exploit', 'persistence',
                'exfiltration', 'lateral_movement', 'privilege_escalation',
                'credential_dump', 'network_scan', 'service_enum', 'banner_grab',
                'tech_fingerprint', 'cms_scan', 'plugin_scan', 'config_audit'
            ]
            
            for i, attack_type in enumerate(attack_types):
                if any(attack_type in att.lower() for att in attack_state.attempted_attacks):
                    state_vector[50 + i] = 1.0
            
            # Encode current phase (80-88)
            phases = ['recon', 'scanning', 'enumeration', 'exploitation', 'post_exploit', 'persistence', 'exfiltration', 'cleanup']
            for i, phase in enumerate(phases):
                if attack_state.current_phase.lower() == phase:
                    state_vector[80 + i] = 1.0
            
            # Encode response patterns (88-108)
            if attack_state.target_responses:
                response_codes = [resp.get('status_code', 0) for resp in attack_state.target_responses]
                avg_response_time = np.mean([resp.get('response_time', 0) for resp in attack_state.target_responses])
                error_rate = sum(1 for code in response_codes if code >= 400) / len(response_codes)
                
                state_vector[88] = np.mean(response_codes) / 1000.0  # Normalize status codes
                state_vector[89] = avg_response_time / 10.0  # Normalize response time
                state_vector[90] = error_rate
                
                # Encode specific response codes
                important_codes = [200, 301, 302, 400, 401, 403, 404, 500, 502, 503]
                for i, code in enumerate(important_codes):
                    state_vector[91 + i] = response_codes.count(code) / len(response_codes)
            
            # Success rate metrics (101-108)
            if attack_state.success_history:
                recent_success_rate = sum(attack_state.success_history[-10:]) / min(10, len(attack_state.success_history))
                overall_success_rate = sum(attack_state.success_history) / len(attack_state.success_history)
                state_vector[101] = recent_success_rate
                state_vector[102] = overall_success_rate
            
            # Remaining slots for future expansion (108-128)
            state_vector[108] = random.random() * 0.1  # Noise for exploration
            
            return state_vector
            
        except Exception as e:
            self.logger.error(f"Failed to encode state: {str(e)}")
            return np.zeros(self.state_size)
    
    def select_action(self, state: AttackState, available_actions: List[AttackAction] = None) -> AttackAction:
        """Select next attack action using epsilon-greedy strategy"""
        if not torch:
            # Fallback to heuristic selection
            return self._heuristic_action_selection(state, available_actions)
        
        if available_actions is None:
            available_actions = self.action_space
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(available_actions)
        else:
            # Exploitation: use Q-network
            state_vector = self.encode_state(state)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            # Map Q-values to available actions
            action_indices = [self._get_action_index(action) for action in available_actions]
            available_q_values = q_values[0][action_indices]
            best_action_idx = action_indices[torch.argmax(available_q_values).item()]
            
            return available_actions[action_indices.index(best_action_idx)]
    
    def _get_action_index(self, action: AttackAction) -> int:
        """Get the index of an action in the action space"""
        for i, a in enumerate(self.action_space):
            if a.action_id == action.action_id:
                return i
        return 0  # Default to first action if not found
    
    def _heuristic_action_selection(self, state: AttackState, available_actions: List[AttackAction]) -> AttackAction:
        """Fallback heuristic action selection when torch is not available"""
        if not available_actions:
            available_actions = self.action_space
        
        # Simple heuristic: prefer actions that haven't been tried yet
        untried_actions = [
            action for action in available_actions
            if action.attack_type not in state.attempted_attacks
        ]
        
        if untried_actions:
            # Among untried actions, prefer lower risk ones early in the mission
            if state.time_elapsed < 1800:  # First 30 minutes
                untried_actions.sort(key=lambda x: x.risk_level)
            return untried_actions[0]
        else:
            # All actions tried, select randomly
            return random.choice(available_actions)
    
    def calculate_reward(self, action: AttackAction, result: Dict[str, Any], 
                        state_before: AttackState, state_after: AttackState) -> AttackReward:
        """Calculate reward for an attack action"""
        base_reward = 0.0
        success_bonus = 0.0
        stealth_bonus = 0.0
        efficiency_bonus = 0.0
        innovation_bonus = 0.0
        detection_penalty = 0.0
        time_penalty = 0.0
        
        # Base reward based on attack outcome
        if result.get('success', False):
            if result.get('critical_impact', False):
                base_reward = self.reward_config['successful_exploit']
                success_bonus = 50.0  # Extra bonus for critical success
            elif result.get('partial_success', False):
                base_reward = self.reward_config['partial_success']
            else:
                base_reward = self.reward_config['information_disclosure']
        else:
            base_reward = self.reward_config['failed_attack']
        
        # Stealth bonus (low detection probability)
        detection_prob = result.get('detection_probability', 0.5)
        if detection_prob < 0.2:
            stealth_bonus = self.reward_config['stealth_bonus']
        elif detection_prob > 0.8:
            detection_penalty = self.reward_config['detection_trigger']
        
        # Efficiency bonus (fast response time)
        response_time = result.get('response_time', 5.0)
        if response_time < 2.0:
            efficiency_bonus = self.reward_config['time_efficiency_bonus']
        elif response_time > 10.0:
            time_penalty = -10.0
        
        # Innovation bonus (novel payload or technique)
        if result.get('novel_technique', False):
            innovation_bonus = self.reward_config['innovation_bonus']
        
        # Chained exploit bonus
        if len(state_after.success_history) > len(state_before.success_history):
            if len([s for s in state_after.success_history[-3:] if s]) >= 2:
                success_bonus += self.reward_config['chained_exploit_bonus']
        
        # Time penalty for long missions
        time_elapsed = state_after.time_elapsed - state_before.time_elapsed
        if time_elapsed > 1800:  # 30 minutes
            time_penalty -= (time_elapsed - 1800) / 3600 * 10  # Penalty increases with time
        
        total_reward = (base_reward + success_bonus + stealth_bonus + 
                       efficiency_bonus + innovation_bonus - 
                       abs(detection_penalty) - abs(time_penalty))
        
        return AttackReward(
            base_reward=base_reward,
            success_bonus=success_bonus,
            stealth_bonus=stealth_bonus,
            efficiency_bonus=efficiency_bonus,
            innovation_bonus=innovation_bonus,
            detection_penalty=detection_penalty,
            time_penalty=time_penalty,
            total_reward=total_reward
        )
    
    def store_experience(self, state: AttackState, action: AttackAction, 
                        reward: AttackReward, next_state: AttackState, done: bool):
        """Store experience in replay memory"""
        state_vector = self.encode_state(state)
        next_state_vector = self.encode_state(next_state)
        action_index = self._get_action_index(action)
        
        experience = (
            state_vector,
            action_index,
            reward.total_reward,
            next_state_vector,
            done
        )
        
        self.memory.append(experience)
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step on the DQN"""
        if not torch or len(self.memory) < self.batch_size:
            return {'loss': 0.0, 'q_value_mean': 0.0}
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            'loss': loss.item(),
            'q_value_mean': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }
    
    def update_target_network(self):
        """Update target network with weights from main network"""
        if torch:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    async def execute_rl_mission(self, mission_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mission using reinforcement learning"""
        mission_id = mission_config.get('mission_id')
        target = mission_config.get('target')
        max_episodes = mission_config.get('max_episodes', 10)
        
        mission_results = {
            'mission_id': mission_id,
            'target': target,
            'started_at': datetime.now().isoformat(),
            'episodes': [],
            'training_metrics': {},
            'final_performance': {},
            'rl_improvements': {}
        }
        
        try:
            self.logger.info(f"Starting RL mission {mission_id} against {target}")
            
            for episode in range(max_episodes):
                episode_result = await self._execute_rl_episode(
                    episode, mission_config
                )
                mission_results['episodes'].append(episode_result)
                
                # Train after each episode
                if len(self.memory) >= self.batch_size:
                    for _ in range(10):  # Multiple training steps per episode
                        train_metrics = self.train_step()
                        for key, value in train_metrics.items():
                            if key not in mission_results['training_metrics']:
                                mission_results['training_metrics'][key] = []
                            mission_results['training_metrics'][key].append(value)
                
                # Update target network periodically
                if episode % 5 == 0:
                    self.update_target_network()
                
                self.logger.info(f"Episode {episode + 1}/{max_episodes} completed")
            
            # Calculate final performance metrics
            mission_results['final_performance'] = self._calculate_rl_performance(
                mission_results['episodes']
            )
            
            mission_results['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            mission_results['error'] = str(e)
            mission_results['failed_at'] = datetime.now().isoformat()
            self.logger.error(f"RL mission {mission_id} failed: {str(e)}")
        
        return mission_results
    
    async def _execute_rl_episode(self, episode_num: int, mission_config: Dict) -> Dict:
        """Execute a single RL episode"""
        episode_results = {
            'episode': episode_num,
            'actions_taken': [],
            'rewards_received': [],
            'total_reward': 0.0,
            'success_count': 0,
            'detection_events': 0
        }
        
        # Initialize state
        current_state = AttackState(
            target_info={'target': mission_config.get('target')},
            discovered_services=[],
            identified_vulns=[],
            attempted_attacks=[],
            success_history=[],
            current_phase='recon',
            time_elapsed=0.0,
            detection_risk=0.0,
            available_tools=['nmap', 'sqlmap', 'gobuster', 'custom'],
            target_responses=[]
        )
        
        max_actions = mission_config.get('max_actions_per_episode', 20)
        
        for step in range(max_actions):
            # Select action
            action = self.select_action(current_state)
            
            # Execute action (mock execution for now)
            action_result = await self._execute_rl_action(action, current_state)
            
            # Update state
            next_state = self._update_state(current_state, action, action_result)
            
            # Calculate reward
            reward = self.calculate_reward(action, action_result, current_state, next_state)
            
            # Store experience
            done = (step == max_actions - 1) or action_result.get('mission_complete', False)
            self.store_experience(current_state, action, reward, next_state, done)
            
            # Update episode results
            episode_results['actions_taken'].append({
                'action': action.action_id,
                'attack_type': action.attack_type,
                'success': action_result.get('success', False)
            })
            episode_results['rewards_received'].append(reward.total_reward)
            episode_results['total_reward'] += reward.total_reward
            
            if action_result.get('success', False):
                episode_results['success_count'] += 1
            
            if action_result.get('detection_probability', 0) > 0.8:
                episode_results['detection_events'] += 1
            
            # Move to next state
            current_state = next_state
            
            if done:
                break
        
        return episode_results
    
    async def _execute_rl_action(self, action: AttackAction, state: AttackState) -> Dict:
        """Execute an attack action (mock implementation)"""
        # This would integrate with the actual attack engine
        # For now, simulate results based on action characteristics
        
        base_success_prob = 0.3
        
        # Adjust success probability based on state
        if action.attack_type in state.attempted_attacks:
            base_success_prob *= 0.7  # Repeated attacks less likely to succeed
        
        if len(state.success_history) > 0:
            recent_success_rate = sum(state.success_history[-5:]) / min(5, len(state.success_history))
            base_success_prob += recent_success_rate * 0.2
        
        # Simulate attack execution
        success = random.random() < base_success_prob
        response_time = random.uniform(0.5, 5.0)
        detection_prob = random.uniform(0.1, 0.6) + (action.risk_level * 0.3)
        
        return {
            'success': success,
            'partial_success': random.random() < 0.2 if not success else False,
            'critical_impact': random.random() < 0.1 if success else False,
            'response_time': response_time,
            'detection_probability': min(detection_prob, 1.0),
            'novel_technique': random.random() < 0.05,
            'mission_complete': success and random.random() < 0.3
        }
    
    def _update_state(self, current_state: AttackState, action: AttackAction, 
                     result: Dict) -> AttackState:
        """Update attack state based on action result"""
        new_state = AttackState(
            target_info=current_state.target_info.copy(),
            discovered_services=current_state.discovered_services.copy(),
            identified_vulns=current_state.identified_vulns.copy(),
            attempted_attacks=current_state.attempted_attacks + [action.attack_type],
            success_history=current_state.success_history + [result.get('success', False)],
            current_phase=current_state.current_phase,
            time_elapsed=current_state.time_elapsed + result.get('response_time', 1.0),
            detection_risk=min(current_state.detection_risk + result.get('detection_probability', 0.1), 1.0),
            available_tools=current_state.available_tools,
            target_responses=current_state.target_responses + [result]
        )
        
        # Update phase based on progress
        if len(new_state.success_history) > 3:
            new_state.current_phase = 'exploitation'
        elif len(new_state.attempted_attacks) > 5:
            new_state.current_phase = 'enumeration'
        
        return new_state
    
    def _calculate_rl_performance(self, episodes: List[Dict]) -> Dict:
        """Calculate performance metrics for RL mission"""
        if not episodes:
            return {}
        
        total_rewards = [ep['total_reward'] for ep in episodes]
        success_counts = [ep['success_count'] for ep in episodes]
        detection_events = [ep['detection_events'] for ep in episodes]
        
        return {
            'average_reward': np.mean(total_rewards),
            'reward_improvement': total_rewards[-1] - total_rewards[0] if len(total_rewards) > 1 else 0,
            'average_success_rate': np.mean(success_counts) / 20,  # Assuming max 20 actions
            'average_detection_events': np.mean(detection_events),
            'learning_stability': np.std(total_rewards[-5:]) if len(total_rewards) >= 5 else float('inf'),
            'final_epsilon': self.epsilon
        }
    
    def save_model(self, model_path: str):
        """Save the trained RL model"""
        if torch:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'training_metrics': dict(self.training_metrics)
            }, model_path)
            self.logger.info(f"RL model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained RL model"""
        if torch and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.training_metrics = defaultdict(list, checkpoint.get('training_metrics', {}))
            self.logger.info(f"RL model loaded from {model_path}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current RL performance metrics"""
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'recent_performance': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
            'learning_progress': self._calculate_learning_progress(),
            'exploration_rate': self.epsilon,
            'memory_utilization': len(self.memory) / self.memory_size
        }
    
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress based on reward improvement"""
        if len(self.episode_rewards) < 20:
            return 0.0
        
        early_performance = np.mean(self.episode_rewards[:10])
        recent_performance = np.mean(self.episode_rewards[-10:])
        
        if early_performance == 0:
            return 1.0 if recent_performance > 0 else 0.0
        
        improvement = (recent_performance - early_performance) / abs(early_performance)
        return max(0.0, min(1.0, improvement))  # Clamp between 0 and 1-