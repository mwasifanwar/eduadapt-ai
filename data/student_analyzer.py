import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta

class StudentAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def generate_sample_students(self, num_students: int = 100) -> List[Dict[str, Any]]:
        students = []
        learning_styles = ['visual', 'auditory', 'kinesthetic', 'reading_writing']
        
        for i in range(num_students):
            student = {
                'student_id': f'student_{i+1:03d}',
                'learning_style': np.random.choice(learning_styles),
                'initial_knowledge': np.random.uniform(0.1, 0.3, 10),
                'engagement_trend': np.random.uniform(0.3, 0.8),
                'preferred_duration': np.random.normal(300, 60),
                'performance_variance': np.random.uniform(0.1, 0.3)
            }
            students.append(student)
        
        return students
    
    def simulate_learning_session(self, student_profile: Dict[str, Any], 
                                content_data: Dict[str, Any]) -> Dict[str, Any]:
        base_performance = np.random.normal(0.6, 0.2)
        
        style_match = self.calculate_style_match(
            content_data['type'],
            student_profile['learning_style']
        )
        
        difficulty_factor = 1 - abs(content_data['difficulty'] - 0.5)
        engagement_factor = student_profile['engagement_trend']
        
        performance = base_performance * (0.6 + 0.2 * style_match + 0.1 * difficulty_factor + 0.1 * engagement_factor)
        performance = np.clip(performance, 0, 1)
        
        base_time = content_data['duration']
        time_variation = np.random.normal(1, 0.2)
        time_spent = base_time * time_variation
        
        interaction_count = max(1, int(np.random.poisson(3)))
        
        return {
            'performance': performance,
            'time_spent': time_spent,
            'interaction_count': interaction_count,
            'content_type': content_data['type'],
            'concepts': content_data['concepts'],
            'timestamp': datetime.now()
        }
    
    def calculate_style_match(self, content_type: str, learning_style: str) -> float:
        style_preferences = {
            'visual': ['video', 'animation'],
            'auditory': ['audio', 'podcast'],
            'kinesthetic': ['interactive', 'simulation'],
            'reading_writing': ['text', 'article']
        }
        
        if learning_style in style_preferences and content_type in style_preferences[learning_style]:
            return 1.0
        elif content_type == 'mixed':
            return 0.7
        else:
            return 0.3
    
    def analyze_learning_patterns(self, student_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not student_history:
            return {}
        
        performances = [session['performance'] for session in student_history]
        times = [session['time_spent'] for session in student_history]
        concepts_covered = set()
        
        for session in student_history:
            concepts_covered.update(session['concepts'])
        
        avg_performance = np.mean(performances)
        performance_trend = self.calculate_trend(performances)
        efficiency = avg_performance / (np.mean(times) / 60)
        
        return {
            'average_performance': avg_performance,
            'performance_trend': performance_trend,
            'learning_efficiency': efficiency,
            'concepts_covered': len(concepts_covered),
            'total_learning_time': sum(times) / 60,
            'consistency': 1 - np.std(performances)
        }
    
    def calculate_trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope