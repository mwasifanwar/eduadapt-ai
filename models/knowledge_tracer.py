import torch
import torch.nn as nn

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