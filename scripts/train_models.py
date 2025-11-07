import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.student_model import StudentModel
from core.learning_optimizer import LearningOptimizer
from data.content_processor import ContentProcessor
from data.student_analyzer import StudentAnalyzer
from utils.config import Config
import torch

def main():
    config = Config()
    
    print("Initializing EduAdapt AI Training...")
    
    content_processor = ContentProcessor(config.get('content', {}))
    student_analyzer = StudentAnalyzer(config.get('students', {}))
    
    print("Loading educational content...")
    sample_content = content_processor.generate_sample_content()
    
    print("Initializing student model...")
    student_model = StudentModel(config.get('student_model', {}))
    
    concepts = set()
    for content in sample_content:
        concepts.update(content['concepts'])
    
    student_model.concept_mapping = {concept: idx for idx, concept in enumerate(concepts)}
    student_model.initialize_knowledge_tracer(len(concepts))
    
    print("Generating sample students...")
    sample_students = student_analyzer.generate_sample_students(50)
    
    print("Simulating learning interactions...")
    for student in sample_students:
        student_id = student['student_id']
        
        for content in sample_content[:5]:
            session_result = student_analyzer.simulate_learning_session(student, content)
            
            interaction_data = {
                'concept_id': content['concepts'][0] if content['concepts'] else None,
                'performance': session_result['performance'],
                'content_type': content['type'],
                'time_spent': session_result['time_spent'],
                'interaction_count': session_result['interaction_count']
            }
            
            student_model.update_student_profile(student_id, interaction_data)
    
    print("Initializing learning optimizer...")
    state_size = len(concepts) + 4
    action_size = len(sample_content)
    
    learning_optimizer = LearningOptimizer(config.get('optimization', {}))
    learning_optimizer.initialize_agent(state_size, action_size)
    
    print("EduAdapt AI models trained successfully!")
    
    return student_model, learning_optimizer

if __name__ == "__main__":
    student_model, learning_optimizer = main()