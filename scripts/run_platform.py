import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import random
from core.student_model import StudentModel
from core.content_recommender import ContentRecommender
from core.learning_optimizer import LearningOptimizer
from data.content_processor import ContentProcessor
from data.student_analyzer import StudentAnalyzer
from assessment.quiz_generator import QuizGenerator
from assessment.progress_tracker import ProgressTracker
from api.endpoints import EduAdaptAPI
from utils.config import Config

class EduAdaptPlatform:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = Config(config_path)
        
        self.student_model = StudentModel(self.config.get('student_model', {}))
        self.content_recommender = ContentRecommender(self.config.get('content', {}))
        self.learning_optimizer = LearningOptimizer(self.config.get('optimization', {}))
        self.content_processor = ContentProcessor(self.config.get('content', {}))
        self.student_analyzer = StudentAnalyzer(self.config.get('students', {}))
        self.quiz_generator = QuizGenerator(self.config.get('assessment', {}))
        self.progress_tracker = ProgressTracker(self.config.get('progress', {}))
        
        self.initialize_platform()
    
    def initialize_platform(self):
        print("Initializing EduAdapt Platform...")
        
        sample_content = self.content_processor.generate_sample_content()
        for content in sample_content:
            self.content_recommender.add_content(content['id'], content)
        
        self.content_recommender.fit_content_vectors()
        
        concepts = set()
        for content in sample_content:
            concepts.update(content['concepts'])
        
        self.student_model.concept_mapping = {concept: idx for idx, concept in enumerate(concepts)}
        self.student_model.initialize_knowledge_tracer(len(concepts))
        
        state_size = len(concepts) + 4
        action_size = len(sample_content)
        self.learning_optimizer.initialize_agent(state_size, action_size)
        
        print(f"Platform initialized with {len(concepts)} concepts and {len(sample_content)} content items")
    
    def get_personalized_recommendations(self, student_id: str, 
                                       target_concepts: List[str],
                                       max_recommendations: int = 5) -> List[Dict[str, Any]]:
        knowledge_gaps = self.student_model.get_student_knowledge_gap(student_id, target_concepts)
        
        if student_id in self.student_model.student_profiles:
            student_profile = self.student_model.student_profiles[student_id]
        else:
            student_profile = {
                'learning_style': 'reading_writing',
                'engagement_level': 0.5,
                'learning_pace': 1.0
            }
        
        recommendations = self.content_recommender.recommend_content(
            student_profile, knowledge_gaps, max_recommendations
        )
        
        return recommendations
    
    def record_learning_session(self, student_id: str, session_data: Dict[str, Any]):
        self.student_model.update_student_profile(student_id, session_data)
        self.progress_tracker.record_learning_session(student_id, session_data)
    
    def get_student_progress(self, student_id: str) -> Dict[str, Any]:
        insights = self.progress_tracker.get_student_insights(student_id)
        
        if student_id in self.student_model.student_profiles:
            student_profile = self.student_model.student_profiles[student_id]
            insights['current_profile'] = student_profile
        
        return insights

def main():
    platform = EduAdaptPlatform()
    
    print("Starting EduAdapt AI Platform...")
    
    api = EduAdaptAPI(platform)
    
    print("EduAdapt AI Platform is running!")
    print("API available at http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    
    api.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
