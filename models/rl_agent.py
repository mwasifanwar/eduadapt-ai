import torch
import torch.nn as nn

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