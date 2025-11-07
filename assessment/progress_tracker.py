import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

class ProgressTracker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.student_progress = {}
        
    def initialize_student_progress(self, student_id: str):
        if student_id not in self.student_progress:
            self.student_progress[student_id] = {
                'learning_sessions': [],
                'assessment_results': [],
                'knowledge_growth': {},
                'engagement_history': [],
                'learning_path': []
            }
    
    def record_learning_session(self, student_id: str, session_data: Dict[str, Any]):
        self.initialize_student_progress(student_id)
        
        session_record = {
            'timestamp': datetime.now(),
            'content_id': session_data.get('content_id'),
            'concepts': session_data.get('concepts', []),
            'performance': session_data.get('performance', 0),
            'time_spent': session_data.get('time_spent', 0),
            'engagement_level': session_data.get('engagement_level', 0.5)
        }
        
        self.student_progress[student_id]['learning_sessions'].append(session_record)
        
        for concept in session_data.get('concepts', []):
            if concept not in self.student_progress[student_id]['knowledge_growth']:
                self.student_progress[student_id]['knowledge_growth'][concept] = []
            
            self.student_progress[student_id]['knowledge_growth'][concept].append({
                'timestamp': datetime.now(),
                'knowledge_level': session_data.get('performance', 0)
            })
    
    def record_assessment_result(self, student_id: str, assessment_data: Dict[str, Any]):
        self.initialize_student_progress(student_id)
        
        assessment_record = {
            'timestamp': datetime.now(),
            'assessment_type': assessment_data.get('type', 'quiz'),
            'overall_score': assessment_data.get('overall_score', 0),
            'concept_scores': assessment_data.get('concept_scores', {}),
            'weak_concepts': assessment_data.get('weak_concepts', [])
        }
        
        self.student_progress[student_id]['assessment_results'].append(assessment_record)
    
    def calculate_learning_velocity(self, student_id: str, days: int = 30) -> float:
        if student_id not in self.student_progress:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = [
            session for session in self.student_progress[student_id]['learning_sessions']
            if session['timestamp'] > cutoff_date
        ]
        
        if not recent_sessions:
            return 0.0
        
        total_learning_time = sum(session['time_spent'] for session in recent_sessions)
        total_concepts_covered = len(set(
            concept for session in recent_sessions for concept in session['concepts']
        ))
        
        if total_learning_time == 0:
            return 0.0
        
        learning_velocity = total_concepts_covered / (total_learning_time / 3600)
        return learning_velocity
    
    def get_student_insights(self, student_id: str) -> Dict[str, Any]:
        if student_id not in self.student_progress:
            return {}
        
        progress_data = self.student_progress[student_id]
        
        total_sessions = len(progress_data['learning_sessions'])
        total_learning_time = sum(session['time_spent'] for session in progress_data['learning_sessions'])
        avg_performance = np.mean([session['performance'] for session in progress_data['learning_sessions']]) if progress_data['learning_sessions'] else 0
        
        recent_assessments = progress_data['assessment_results'][-5:]
        assessment_trend = self.calculate_assessment_trend(recent_assessments)
        
        learning_velocity = self.calculate_learning_velocity(student_id)
        
        return {
            'total_sessions': total_sessions,
            'total_learning_hours': total_learning_time / 3600,
            'average_performance': avg_performance,
            'learning_velocity': learning_velocity,
            'assessment_trend': assessment_trend,
            'recent_engagement': self.calculate_recent_engagement(student_id),
            'knowledge_gaps': self.identify_knowledge_gaps(student_id)
        }
    
    def calculate_assessment_trend(self, assessments: List[Dict[str, Any]]) -> float:
        if len(assessments) < 2:
            return 0.0
        
        scores = [assessment['overall_score'] for assessment in assessments]
        return (scores[-1] - scores[0]) / len(scores)
    
    def calculate_recent_engagement(self, student_id: str, days: int = 7) -> float:
        if student_id not in self.student_progress:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_sessions = [
            session for session in self.student_progress[student_id]['learning_sessions']
            if session['timestamp'] > cutoff_date
        ]
        
        if not recent_sessions:
            return 0.0
        
        return np.mean([session['engagement_level'] for session in recent_sessions])
    
    def identify_knowledge_gaps(self, student_id: str) -> List[str]:
        if student_id not in self.student_progress:
            return []
        
        weak_concepts = set()
        
        for assessment in self.student_progress[student_id]['assessment_results']:
            weak_concepts.update(assessment.get('weak_concepts', []))
        
        return list(weak_concepts)