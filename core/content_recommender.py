import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentRecommender:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.content_db = {}
        self.concept_graph = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_vectors = None
        self.content_features = []
        
    def add_content(self, content_id: str, content_data: Dict[str, Any]):
        self.content_db[content_id] = content_data
        
        concepts = content_data.get('concepts', [])
        difficulty = content_data.get('difficulty', 0.5)
        content_type = content_data.get('type', 'text')
        duration = content_data.get('duration', 300)
        
        features = {
            'concepts': concepts,
            'difficulty': difficulty,
            'type': content_type,
            'duration': duration,
            'text_content': content_data.get('description', '') + ' ' + content_data.get('title', '')
        }
        
        self.content_features.append(features)
        
        for concept in concepts:
            if concept not in self.concept_graph:
                self.concept_graph[concept] = {'prerequisites': [], 'next_concepts': []}
    
    def build_concept_graph(self, prerequisites: List[Tuple[str, str]]):
        for prereq, concept in prerequisites:
            if concept in self.concept_graph and prereq in self.concept_graph:
                self.concept_graph[concept]['prerequisites'].append(prereq)
                self.concept_graph[prereq]['next_concepts'].append(concept)
    
    def fit_content_vectors(self):
        texts = [feature['text_content'] for feature in self.content_features]
        if texts:
            self.content_vectors = self.vectorizer.fit_transform(texts)
    
    def recommend_content(self, student_profile: Dict[str, Any], 
                         knowledge_gaps: Dict[str, float],
                         max_recommendations: int = 5) -> List[Dict[str, Any]]:
        learning_style = student_profile.get('learning_style', 'reading_writing')
        engagement_level = student_profile.get('engagement_level', 0.5)
        learning_pace = student_profile.get('learning_pace', 1.0)
        
        candidate_scores = {}
        
        for content_id, content_data in self.content_db.items():
            score = self.calculate_content_score(
                content_data, knowledge_gaps, learning_style, engagement_level, learning_pace
            )
            candidate_scores[content_id] = score
        
        sorted_content = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for content_id, score in sorted_content[:max_recommendations]:
            content_data = self.content_db[content_id].copy()
            content_data['recommendation_score'] = score
            content_data['content_id'] = content_id
            recommendations.append(content_data)
        
        return recommendations
    
    def calculate_content_score(self, content_data: Dict[str, Any],
                              knowledge_gaps: Dict[str, float],
                              learning_style: str,
                              engagement_level: float,
                              learning_pace: float) -> float:
        concepts = content_data.get('concepts', [])
        difficulty = content_data.get('difficulty', 0.5)
        content_type = content_data.get('type', 'text')
        duration = content_data.get('duration', 300)
        
        relevance_score = 0.0
        for concept in concepts:
            if concept in knowledge_gaps:
                relevance_score += knowledge_gaps[concept]
        
        if concepts:
            relevance_score /= len(concepts)
        
        style_match_score = self.calculate_style_match(content_type, learning_style)
        difficulty_score = self.calculate_difficulty_match(difficulty, engagement_level)
        duration_score = self.calculate_duration_match(duration, learning_pace, engagement_level)
        
        total_score = (
            relevance_score * 0.5 +
            style_match_score * 0.3 +
            difficulty_score * 0.1 +
            duration_score * 0.1
        )
        
        return total_score
    
    def calculate_style_match(self, content_type: str, learning_style: str) -> float:
        style_mapping = {
            'visual': ['video', 'animation', 'infographic'],
            'auditory': ['audio', 'podcast', 'lecture'],
            'kinesthetic': ['interactive', 'simulation', 'game'],
            'reading_writing': ['text', 'article', 'ebook']
        }
        
        if learning_style in style_mapping and content_type in style_mapping[learning_style]:
            return 1.0
        else:
            return 0.3
    
    def calculate_difficulty_match(self, difficulty: float, engagement_level: float) -> float:
        target_difficulty = 0.3 + (engagement_level * 0.4)
        difficulty_diff = abs(difficulty - target_difficulty)
        return max(0, 1 - difficulty_diff * 2)
    
    def calculate_duration_match(self, duration: float, learning_pace: float, engagement_level: float) -> float:
        ideal_duration = 300 * learning_pace
        if engagement_level > 0.7:
            ideal_duration *= 1.5
        
        duration_ratio = min(duration, ideal_duration) / max(duration, ideal_duration)
        return duration_ratio
    
    def get_prerequisite_path(self, target_concept: str, student_knowledge: Dict[str, float]) -> List[str]:
        if target_concept not in self.concept_graph:
            return []
        
        prerequisites = self.concept_graph[target_concept]['prerequisites']
        learning_path = []
        
        for prereq in prerequisites:
            if student_knowledge.get(prereq, 0) < 0.7:
                learning_path.extend(self.get_prerequisite_path(prereq, student_knowledge))
        
        learning_path.append(target_concept)
        return learning_path