import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

class KnowledgeTracer(nn.Module):
    def __init__(self, num_concepts: int, hidden_size: int = 128):
        super().__init__()
        self.num_concepts = num_concepts
        self.hidden_size = hidden_size
        
        self.concept_embeddings = nn.Embedding(num_concepts, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.knowledge_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_concepts),
            nn.Sigmoid()
        )
        
    def forward(self, concept_seq, performance_seq):
        batch_size, seq_len = concept_seq.shape
        
        concept_emb = self.concept_embeddings(concept_seq)
        performance_emb = performance_seq.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        combined_input = torch.cat([concept_emb, performance_emb], dim=-1)
        
        lstm_out, _ = self.lstm(combined_input)
        knowledge_state = self.knowledge_predictor(lstm_out[:, -1, :])
        
        return knowledge_state

class StudentModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.knowledge_tracer = None
        self.concept_mapping = {}
        self.student_profiles = {}
        
    def initialize_knowledge_tracer(self, num_concepts: int):
        hidden_size = self.config.get('hidden_size', 128)
        self.knowledge_tracer = KnowledgeTracer(num_concepts, hidden_size).to(self.device)
        
    def update_student_profile(self, student_id: str, interaction_data: Dict[str, Any]):
        if student_id not in self.student_profiles:
            self.student_profiles[student_id] = {
                'knowledge_state': np.zeros(len(self.concept_mapping)),
                'learning_style': self.detect_learning_style(interaction_data),
                'engagement_level': 0.5,
                'performance_history': [],
                'preferred_content_types': [],
                'learning_pace': 1.0
            }
        
        profile = self.student_profiles[student_id]
        
        concept_id = interaction_data.get('concept_id')
        performance = interaction_data.get('performance', 0)
        content_type = interaction_data.get('content_type')
        time_spent = interaction_data.get('time_spent', 0)
        
        if concept_id is not None and concept_id in self.concept_mapping:
            concept_idx = self.concept_mapping[concept_id]
            profile['knowledge_state'][concept_idx] = self.update_knowledge(
                profile['knowledge_state'][concept_idx], performance, time_spent
            )
        
        profile['performance_history'].append(performance)
        profile['engagement_level'] = self.calculate_engagement(interaction_data, profile)
        profile['learning_pace'] = self.estimate_learning_pace(interaction_data, profile)
        
        if content_type and content_type not in profile['preferred_content_types']:
            profile['preferred_content_types'].append(content_type)
    
    def update_knowledge(self, current_knowledge: float, performance: float, time_spent: float) -> float:
        learning_rate = self.config.get('learning_rate', 0.1)
        forgetting_rate = self.config.get('forgetting_rate', 0.05)
        
        knowledge_gain = performance * learning_rate * (1 - current_knowledge)
        knowledge_decay = forgetting_rate * current_knowledge
        
        new_knowledge = current_knowledge + knowledge_gain - knowledge_decay
        return np.clip(new_knowledge, 0, 1)
    
    def detect_learning_style(self, interaction_data: Dict[str, Any]) -> str:
        content_type = interaction_data.get('content_type', 'text')
        time_spent = interaction_data.get('time_spent', 0)
        performance = interaction_data.get('performance', 0)
        
        if content_type in ['video', 'animation'] and time_spent > 300:
            return 'visual'
        elif content_type in ['audio', 'podcast'] and performance > 0.7:
            return 'auditory'
        elif content_type in ['interactive', 'simulation']:
            return 'kinesthetic'
        else:
            return 'reading_writing'
    
    def calculate_engagement(self, interaction_data: Dict[str, Any], profile: Dict[str, Any]) -> float:
        time_spent = interaction_data.get('time_spent', 0)
        performance = interaction_data.get('performance', 0)
        interactions = interaction_data.get('interaction_count', 1)
        
        time_score = min(time_spent / 600, 1.0)
        performance_score = performance
        interaction_score = min(interactions / 10, 1.0)
        
        engagement = (time_score * 0.4 + performance_score * 0.4 + interaction_score * 0.2)
        current_engagement = profile.get('engagement_level', 0.5)
        
        return 0.7 * current_engagement + 0.3 * engagement
    
    def estimate_learning_pace(self, interaction_data: Dict[str, Any], profile: Dict[str, Any]) -> float:
        recent_performance = profile['performance_history'][-10:] if len(profile['performance_history']) >= 10 else profile['performance_history']
        if not recent_performance:
            return 1.0
        
        avg_performance = np.mean(recent_performance)
        time_spent = interaction_data.get('time_spent', 300)
        
        if avg_performance > 0.8 and time_spent < 300:
            return 1.5
        elif avg_performance < 0.5 and time_spent > 600:
            return 0.7
        else:
            return 1.0
    
    def get_student_knowledge_gap(self, student_id: str, target_concepts: List[str]) -> Dict[str, float]:
        if student_id not in self.student_profiles:
            return {concept: 1.0 for concept in target_concepts}
        
        profile = self.student_profiles[student_id]
        knowledge_gaps = {}
        
        for concept in target_concepts:
            if concept in self.concept_mapping:
                concept_idx = self.concept_mapping[concept]
                knowledge_gaps[concept] = 1.0 - profile['knowledge_state'][concept_idx]
            else:
                knowledge_gaps[concept] = 1.0
        
        return knowledge_gaps
