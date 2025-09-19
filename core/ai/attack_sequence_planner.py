#!/usr/bin/env python3
"""
Agent DS v2.0 - Attack Sequence Planning
========================================

Advanced planning algorithms for attack sequence optimization:
- Monte Carlo Tree Search for sequence planning
- Genetic Algorithm for sequence evolution
- Dynamic Programming for optimal resource allocation
- Constraint Satisfaction for complex requirements

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

# Import the main sequencer components
from .adaptive_attack_sequencer import (
    AttackAction, AttackResult, AttackPhase, AttackVector, 
    SequenceState, ResourceType
)

@dataclass
class PlanningNode:
    """Node in attack sequence planning tree"""
    state: SequenceState
    parent: Optional['PlanningNode']
    children: List['PlanningNode']
    visits: int = 0
    value: float = 0.0
    action: Optional[AttackAction] = None
    depth: int = 0

class MonteCarloTreeSearch:
    """Monte Carlo Tree Search for attack sequence planning"""
    
    def __init__(self, exploration_weight: float = 1.4):
        self.exploration_weight = exploration_weight
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def search(self, 
                    root_state: SequenceState,
                    available_actions: List[AttackAction],
                    max_iterations: int = 1000,
                    max_depth: int = 10) -> List[AttackAction]:
        """
        Perform MCTS to find optimal attack sequence
        """
        self.logger.info(f"Starting MCTS with {max_iterations} iterations")
        
        root = PlanningNode(state=root_state, parent=None, children=[])
        
        for iteration in range(max_iterations):
            # Selection
            node = self._select(root, max_depth)
            
            # Expansion
            if node.visits > 0 and node.depth < max_depth:
                node = self._expand(node, available_actions)
            
            # Simulation
            reward = await self._simulate(node, available_actions, max_depth - node.depth)
            
            # Backpropagation
            self._backpropagate(node, reward)
            
            if iteration % 100 == 0:
                self.logger.debug(f"MCTS iteration {iteration}, best value: {root.value}")
        
        # Extract best sequence
        sequence = self._extract_best_sequence(root)
        self.logger.info(f"MCTS completed, found sequence of length {len(sequence)}")
        
        return sequence
    
    def _select(self, node: PlanningNode, max_depth: int) -> PlanningNode:
        """Select most promising node using UCB1"""
        current = node
        
        while current.children and current.depth < max_depth:
            if any(child.visits == 0 for child in current.children):
                # Return first unvisited child
                current = next(child for child in current.children if child.visits == 0)
                break
            else:
                # UCB1 selection
                current = max(current.children, key=self._ucb1_value)
        
        return current
    
    def _ucb1_value(self, node: PlanningNode) -> float:
        """Calculate UCB1 value for node selection"""
        if node.visits == 0:
            return float('inf')
        
        exploitation = node.value / node.visits
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.parent.visits) / node.visits
        )
        
        return exploitation + exploration
    
    def _expand(self, node: PlanningNode, available_actions: List[AttackAction]) -> PlanningNode:
        """Expand node with possible actions"""
        # Get applicable actions for current state
        applicable_actions = self._get_applicable_actions(node.state, available_actions)
        
        for action in applicable_actions:
            new_state = self._apply_action(node.state, action)
            child = PlanningNode(
                state=new_state,
                parent=node,
                children=[],
                action=action,
                depth=node.depth + 1
            )
            node.children.append(child)
        
        # Return random child for simulation
        return random.choice(node.children) if node.children else node
    
    async def _simulate(self, 
                       node: PlanningNode, 
                       available_actions: List[AttackAction],
                       max_steps: int) -> float:
        """Simulate random rollout from node"""
        current_state = node.state
        total_reward = 0.0
        
        for step in range(max_steps):
            applicable_actions = self._get_applicable_actions(current_state, available_actions)
            if not applicable_actions:
                break
            
            # Random action selection for simulation
            action = random.choice(applicable_actions)
            
            # Simulate action result
            simulated_result = self._simulate_action_result(action, current_state)
            reward = self._calculate_simulation_reward(action, simulated_result)
            total_reward += reward
            
            # Update state
            current_state = self._apply_action_result(current_state, action, simulated_result)
        
        return total_reward
    
    def _backpropagate(self, node: PlanningNode, reward: float):
        """Backpropagate reward up the tree"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent
    
    def _extract_best_sequence(self, root: PlanningNode) -> List[AttackAction]:
        """Extract best action sequence from tree"""
        sequence = []
        current = root
        
        while current.children:
            # Select child with highest average value
            best_child = max(current.children, 
                           key=lambda c: c.value / c.visits if c.visits > 0 else 0)
            if best_child.action:
                sequence.append(best_child.action)
            current = best_child
        
        return sequence
    
    def _get_applicable_actions(self, 
                              state: SequenceState, 
                              available_actions: List[AttackAction]) -> List[AttackAction]:
        """Get actions applicable in current state"""
        applicable = []
        
        for action in available_actions:
            # Check if action is applicable based on phase and dependencies
            if self._is_action_applicable(action, state):
                applicable.append(action)
        
        return applicable
    
    def _is_action_applicable(self, action: AttackAction, state: SequenceState) -> bool:
        """Check if action is applicable in current state"""
        # Check phase progression
        phase_order = list(AttackPhase)
        current_phase_idx = phase_order.index(state.current_phase)
        action_phase_idx = phase_order.index(action.phase)
        
        # Allow current phase and next phases
        if action_phase_idx < current_phase_idx:
            return False
        
        # Check dependencies
        completed_actions = [result.action for result in state.completed_actions]
        for dependency in action.dependencies:
            if not any(act.vector.value == dependency for act in completed_actions):
                return False
        
        # Check resource availability
        for resource_type, required in action.resource_requirements.items():
            available = state.available_resources.get(resource_type, 0.0)
            if required > available:
                return False
        
        return True
    
    def _apply_action(self, state: SequenceState, action: AttackAction) -> SequenceState:
        """Apply action to state (returns new state)"""
        new_state = SequenceState(
            current_phase=state.current_phase,
            completed_actions=state.completed_actions.copy(),
            pending_actions=state.pending_actions.copy(),
            available_resources=state.available_resources.copy(),
            target_context=state.target_context.copy(),
            success_rate=state.success_rate,
            total_time_elapsed=state.total_time_elapsed,
            phase_completion=state.phase_completion.copy()
        )
        
        # Update pending actions
        if action in new_state.pending_actions:
            new_state.pending_actions.remove(action)
        
        return new_state
    
    def _simulate_action_result(self, action: AttackAction, state: SequenceState) -> AttackResult:
        """Simulate the result of an action"""
        # Simple simulation based on action properties
        success_prob = action.success_probability
        
        # Add some randomness and context-based adjustments
        if state.target_context.get('waf_detected'):
            success_prob *= 0.7
        
        success = random.random() < success_prob
        
        return AttackResult(
            action=action,
            success=success,
            response_time=random.uniform(0.1, 2.0),
            status_code=200 if success else random.choice([403, 404, 500]),
            response_size=random.randint(100, 5000),
            confidence_score=success_prob if success else 1.0 - success_prob
        )
    
    def _calculate_simulation_reward(self, action: AttackAction, result: AttackResult) -> float:
        """Calculate reward for simulation"""
        base_reward = 1.0 if result.success else -0.1
        
        # Bonus for high-priority actions
        priority_bonus = action.priority * 0.5
        
        # Penalty for slow actions
        time_penalty = result.response_time * -0.1
        
        return base_reward + priority_bonus + time_penalty
    
    def _apply_action_result(self, 
                           state: SequenceState, 
                           action: AttackAction, 
                           result: AttackResult) -> SequenceState:
        """Apply action result to state"""
        new_state = self._apply_action(state, action)
        
        # Add to completed actions
        new_state.completed_actions.append(result)
        
        # Update success rate
        total_actions = len(new_state.completed_actions)
        successful_actions = sum(1 for r in new_state.completed_actions if r.success)
        new_state.success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
        
        # Update time elapsed
        new_state.total_time_elapsed += result.response_time
        
        # Update phase completion
        phase_actions = [r for r in new_state.completed_actions if r.action.phase == action.phase]
        # Simple completion metric based on action count
        completion = min(1.0, len(phase_actions) / 5.0)  # Assume 5 actions per phase
        new_state.phase_completion[action.phase] = completion
        
        # Advance phase if current phase is sufficiently complete
        if completion >= 0.8:
            phase_order = list(AttackPhase)
            current_idx = phase_order.index(new_state.current_phase)
            if current_idx < len(phase_order) - 1:
                new_state.current_phase = phase_order[current_idx + 1]
        
        return new_state

class GeneticAlgorithmSequencer:
    """Genetic Algorithm for attack sequence evolution"""
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def evolve_sequence(self, 
                            available_actions: List[AttackAction],
                            target_context: Dict[str, Any],
                            generations: int = 100) -> List[AttackAction]:
        """
        Evolve optimal attack sequence using genetic algorithm
        """
        self.logger.info(f"Starting GA evolution for {generations} generations")
        
        # Initialize population
        population = self._initialize_population(available_actions)
        
        best_fitness_history = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [
                await self._evaluate_fitness(sequence, target_context)
                for sequence in population
            ]
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            if generation % 20 == 0:
                self.logger.debug(f"Generation {generation}, best fitness: {best_fitness:.3f}")
            
            # Selection
            selected = self._select_parents(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            
            # Keep elite
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], 
                                 reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, available_actions)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best sequence
        final_fitness = [
            await self._evaluate_fitness(sequence, target_context)
            for sequence in population
        ]
        best_idx = max(range(len(final_fitness)), key=lambda i: final_fitness[i])
        
        self.logger.info(f"GA evolution completed, best fitness: {final_fitness[best_idx]:.3f}")
        return population[best_idx]
    
    def _initialize_population(self, available_actions: List[AttackAction]) -> List[List[AttackAction]]:
        """Initialize random population of attack sequences"""
        population = []
        
        for _ in range(self.population_size):
            # Random sequence length
            sequence_length = random.randint(3, min(15, len(available_actions)))
            
            # Random selection of actions
            sequence = random.sample(available_actions, sequence_length)
            
            # Sort by phase to maintain logical order
            sequence.sort(key=lambda a: (list(AttackPhase).index(a.phase), random.random()))
            
            population.append(sequence)
        
        return population
    
    async def _evaluate_fitness(self, 
                               sequence: List[AttackAction],
                               target_context: Dict[str, Any]) -> float:
        """Evaluate fitness of attack sequence"""
        fitness = 0.0
        
        # Phase progression bonus
        phases_covered = set(action.phase for action in sequence)
        fitness += len(phases_covered) * 10.0
        
        # Success probability
        total_success_prob = sum(action.success_probability for action in sequence)
        fitness += total_success_prob
        
        # Time efficiency
        total_time = sum(action.estimated_time for action in sequence)
        if total_time > 0:
            fitness += 100.0 / total_time  # Inverse time bonus
        
        # Priority weighting
        priority_score = sum(action.priority for action in sequence)
        fitness += priority_score * 2.0
        
        # Diversity bonus
        vector_diversity = len(set(action.vector for action in sequence))
        fitness += vector_diversity * 5.0
        
        # Dependency satisfaction
        satisfied_deps = 0
        completed_vectors = set()
        
        for action in sequence:
            for dep in action.dependencies:
                if dep in [v.value for v in completed_vectors]:
                    satisfied_deps += 1
            completed_vectors.add(action.vector)
        
        fitness += satisfied_deps * 3.0
        
        return fitness
    
    def _select_parents(self, 
                       population: List[List[AttackAction]], 
                       fitness_scores: List[float]) -> List[List[AttackAction]]:
        """Select parents for crossover using tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population) // 2):
            # Tournament selection
            tournament_indices = random.sample(range(len(population)), tournament_size)
            winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[winner_idx])
        
        return selected
    
    def _crossover(self, 
                  parent1: List[AttackAction], 
                  parent2: List[AttackAction]) -> List[AttackAction]:
        """Perform crossover between two parent sequences"""
        # Order crossover (OX)
        min_length = min(len(parent1), len(parent2))
        if min_length < 2:
            return parent1.copy()
        
        # Select crossover points
        start = random.randint(0, min_length - 1)
        end = random.randint(start + 1, min_length)
        
        # Create child with segment from parent1
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions with actions from parent2
        child_actions = set(child[start:end])
        remaining_actions = [action for action in parent2 if action not in child_actions]
        
        j = 0
        for i in range(len(child)):
            if child[i] is None and j < len(remaining_actions):
                child[i] = remaining_actions[j]
                j += 1
        
        # Remove None values
        child = [action for action in child if action is not None]
        
        return child
    
    def _mutate(self, 
               sequence: List[AttackAction], 
               available_actions: List[AttackAction]) -> List[AttackAction]:
        """Mutate attack sequence"""
        mutated = sequence.copy()
        
        mutation_type = random.choice(['swap', 'insert', 'remove', 'replace'])
        
        if mutation_type == 'swap' and len(mutated) >= 2:
            # Swap two actions
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        elif mutation_type == 'insert':
            # Insert random action
            new_action = random.choice(available_actions)
            insert_pos = random.randint(0, len(mutated))
            mutated.insert(insert_pos, new_action)
        
        elif mutation_type == 'remove' and len(mutated) > 1:
            # Remove random action
            remove_pos = random.randint(0, len(mutated) - 1)
            mutated.pop(remove_pos)
        
        elif mutation_type == 'replace' and mutated:
            # Replace random action
            replace_pos = random.randint(0, len(mutated) - 1)
            new_action = random.choice(available_actions)
            mutated[replace_pos] = new_action
        
        return mutated

class DynamicProgrammingOptimizer:
    """Dynamic Programming for optimal resource allocation"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memo = {}
    
    def optimize_resource_allocation(self, 
                                   actions: List[AttackAction],
                                   resources: Dict[ResourceType, float],
                                   time_budget: float) -> Tuple[List[AttackAction], Dict[str, float]]:
        """
        Optimize resource allocation using dynamic programming
        """
        self.logger.info(f"Optimizing allocation for {len(actions)} actions")
        
        # Convert to DP problem
        items = []
        for i, action in enumerate(actions):
            time_cost = action.estimated_time
            value = action.priority * action.success_probability
            items.append((time_cost, value, i))
        
        # Solve knapsack problem
        selected_indices = self._knapsack_dp(items, time_budget)
        selected_actions = [actions[i] for i in selected_indices]
        
        # Calculate resource allocation
        allocation = self._allocate_resources_dp(selected_actions, resources)
        
        self.logger.info(f"Selected {len(selected_actions)} actions for execution")
        return selected_actions, allocation
    
    def _knapsack_dp(self, 
                    items: List[Tuple[float, float, int]], 
                    capacity: float) -> List[int]:
        """Solve knapsack problem using dynamic programming"""
        n = len(items)
        # Discretize capacity for DP table
        capacity_int = int(capacity * 10)  # 0.1 second precision
        
        # DP table
        dp = [[0 for _ in range(capacity_int + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            weight, value, _ = items[i - 1]
            weight_int = int(weight * 10)
            
            for w in range(capacity_int + 1):
                if weight_int <= w:
                    dp[i][w] = max(dp[i - 1][w], 
                                  dp[i - 1][w - weight_int] + value)
                else:
                    dp[i][w] = dp[i - 1][w]
        
        # Backtrack to find selected items
        selected = []
        w = capacity_int
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(items[i - 1][2])  # Add item index
                weight_int = int(items[i - 1][0] * 10)
                w -= weight_int
        
        return selected
    
    def _allocate_resources_dp(self, 
                             actions: List[AttackAction],
                             available_resources: Dict[ResourceType, float]) -> Dict[str, float]:
        """Allocate resources using DP approach"""
        allocation = {}
        
        for resource_type, total_available in available_resources.items():
            # Calculate total requirements
            total_required = sum(
                action.resource_requirements.get(resource_type, 0.0)
                for action in actions
            )
            
            if total_required <= total_available:
                # Enough resources for all
                for action in actions:
                    action_id = f"{action.vector.value}_{hash(action.target_endpoint)}"
                    required = action.resource_requirements.get(resource_type, 0.0)
                    allocation[f"{action_id}_{resource_type.value}"] = required
            else:
                # Need to allocate proportionally
                for action in actions:
                    action_id = f"{action.vector.value}_{hash(action.target_endpoint)}"
                    required = action.resource_requirements.get(resource_type, 0.0)
                    proportion = required / total_required if total_required > 0 else 0.0
                    allocated = total_available * proportion
                    allocation[f"{action_id}_{resource_type.value}"] = allocated
        
        return allocation

# Export classes
__all__ = [
    'MonteCarloTreeSearch', 
    'GeneticAlgorithmSequencer', 
    'DynamicProgrammingOptimizer',
    'PlanningNode'
]