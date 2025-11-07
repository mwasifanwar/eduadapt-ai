import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
import random

class RLLearningAgent(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        policy_logits = self.policy_network(state)
        value = self.value_network(state)
        return policy_logits, value

class LearningOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = None
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        
    def initialize_agent(self, state_size: int, action_size: int):
        hidden_size = self.config.get('hidden_size', 128)
        self.agent = RLLearningAgent(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate)
    
    def select_learning_action(self, state: np.ndarray, available_actions: List[int], 
                             exploration_rate: float = 0.1) -> int:
        if random.random() < exploration_rate:
            return random.choice(available_actions)
        
        self.agent.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, _ = self.agent(state_tensor)
            
            action_probs = torch.softmax(policy_logits, dim=1)
            available_probs = action_probs[0, available_actions]
            
            selected_idx = torch.argmax(available_probs).item()
            return available_actions[selected_idx]
    
    def update_agent(self, states: List[np.ndarray], actions: List[int], 
                    rewards: List[float], next_states: List[np.ndarray]):
        self.agent.train()
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        
        current_policy_logits, current_values = self.agent(states_tensor)
        _, next_values = self.agent(next_states_tensor)
        
        target_values = rewards_tensor + self.gamma * next_values.squeeze()
        advantage = target_values - current_values.squeeze()
        
        action_probs = torch.softmax(current_policy_logits, dim=1)
        selected_action_probs = action_probs[range(len(actions)), actions_tensor]
        
        policy_loss = -torch.log(selected_action_probs) * advantage.detach()
        value_loss = torch.nn.functional.mse_loss(current_values.squeeze(), target_values.detach())
        
        total_loss = policy_loss.mean() + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def calculate_learning_reward(self, student_profile: Dict[str, Any],
                                content_data: Dict[str, Any],
                                performance: float,
                                time_spent: float) -> float:
        knowledge_gain = performance
        engagement_bonus = student_profile.get('engagement_level', 0.5)
        efficiency_bonus = max(0, 1 - (time_spent / 600))
        
        style_match = self.calculate_style_reward(
            content_data.get('type', 'text'),
            student_profile.get('learning_style', 'reading_writing')
        )
        
        difficulty_match = self.calculate_difficulty_reward(
            content_data.get('difficulty', 0.5),
            student_profile.get('engagement_level', 0.5)
        )
        
        total_reward = (
            knowledge_gain * 0.5 +
            engagement_bonus * 0.2 +
            efficiency_bonus * 0.1 +
            style_match * 0.1 +
            difficulty_match * 0.1
        )
        
        return total_reward
    
    def calculate_style_reward(self, content_type: str, learning_style: str) -> float:
        style_matches = {
            'visual': ['video', 'animation'],
            'auditory': ['audio', 'podcast'],
            'kinesthetic': ['interactive', 'simulation'],
            'reading_writing': ['text', 'article']
        }
        
        for style, preferred_types in style_matches.items():
            if learning_style == style and content_type in preferred_types:
                return 1.0
        return 0.0
    
    def calculate_difficulty_reward(self, difficulty: float, engagement: float) -> float:
        optimal_difficulty = 0.3 + (engagement * 0.4)
        difficulty_diff = abs(difficulty - optimal_difficulty)
        return max(0, 1 - difficulty_diff * 2)
