import json
import pandas as pd
from typing import Dict, List, Any
import numpy as np

class ContentProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.content_db = {}
        
    def load_educational_content(self, content_file: str):
        with open(content_file, 'r') as f:
            content_data = json.load(f)
        
        for content in content_data:
            content_id = content['id']
            self.content_db[content_id] = {
                'title': content['title'],
                'description': content.get('description', ''),
                'concepts': content['concepts'],
                'difficulty': content.get('difficulty', 0.5),
                'type': content.get('type', 'text'),
                'duration': content.get('duration', 300),
                'prerequisites': content.get('prerequisites', [])
            }
    
    def generate_sample_content(self):
        sample_content = [
            {
                'id': 'math_basics_1',
                'title': 'Introduction to Algebra',
                'description': 'Learn basic algebraic concepts and equations',
                'concepts': ['algebra', 'variables', 'equations'],
                'difficulty': 0.3,
                'type': 'video',
                'duration': 600,
                'prerequisites': []
            },
            {
                'id': 'math_basics_2',
                'title': 'Solving Linear Equations',
                'description': 'Practice solving simple linear equations',
                'concepts': ['algebra', 'linear_equations'],
                'difficulty': 0.5,
                'type': 'interactive',
                'duration': 450,
                'prerequisites': ['math_basics_1']
            },
            {
                'id': 'science_physics_1',
                'title': 'Newton Laws of Motion',
                'description': 'Understanding the fundamental laws of motion',
                'concepts': ['physics', 'motion', 'newton_laws'],
                'difficulty': 0.4,
                'type': 'animation',
                'duration': 480,
                'prerequisites': []
            }
        ]
        
        for content in sample_content:
            self.content_db[content['id']] = content
        
        return sample_content
    
    def get_content_by_concept(self, concept: str) -> List[Dict[str, Any]]:
        matching_content = []
        for content_id, content_data in self.content_db.items():
            if concept in content_data['concepts']:
                matching_content.append(content_data)
        return matching_content
    
    def analyze_content_coverage(self, concepts: List[str]) -> Dict[str, List[str]]:
        coverage = {}
        for concept in concepts:
            coverage[concept] = []
            for content_id, content_data in self.content_db.items():
                if concept in content_data['concepts']:
                    coverage[concept].append(content_id)
        return coverage