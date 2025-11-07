import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import pandas as pd
from datetime import datetime, timedelta
from data.student_analyzer import StudentAnalyzer
from data.content_processor import ContentProcessor
from assessment.quiz_generator import QuizGenerator
from utils.config import Config

def simulate_learning_journey(num_students: int = 10, days: int = 30):
    config = Config()
    
    student_analyzer = StudentAnalyzer(config.get('students', {}))
    content_processor = ContentProcessor(config.get('content', {}))
    quiz_generator = QuizGenerator(config.get('assessment', {}))
    
    print(f"Simulating {num_students} students over {days} days...")
    
    students = student_analyzer.generate_sample_students(num_students)
    content_items = content_processor.generate_sample_content()
    
    learning_data = []
    
    for student in students:
        student_id = student['student_id']
        current_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            daily_sessions = random.randint(1, 3)
            
            for session in range(daily_sessions):
                content = random.choice(content_items)
                session_result = student_analyzer.simulate_learning_session(student, content)
                
                learning_record = {
                    'student_id': student_id,
                    'date': current_date,
                    'content_id': content['id'],
                    'content_title': content['title'],
                    'concepts': content['concepts'],
                    'performance': session_result['performance'],
                    'time_spent': session_result['time_spent'],
                    'learning_style': student['learning_style']
                }
                
                learning_data.append(learning_record)
            
            if day % 7 == 0:
                quiz_concepts = random.sample(content['concepts'], min(3, len(content['concepts'])))
                quiz = quiz_generator.generate_adaptive_quiz(student, quiz_concepts, 5)
                
                if quiz:
                    student_answers = {i: random.choice(['A', 'B', 'C', 'D']) for i in range(len(quiz))}
                    quiz_result = quiz_generator.evaluate_quiz_performance(quiz, student_answers)
                    
                    quiz_record = {
                        'student_id': student_id,
                        'date': current_date,
                        'assessment_type': 'weekly_quiz',
                        'score': quiz_result['overall_score'],
                        'concepts_tested': quiz_concepts
                    }
                    
                    learning_data.append(quiz_record)
            
            current_date += timedelta(days=1)
    
    df_learning = pd.DataFrame(learning_data)
    
    print("\n--- Learning Simulation Summary ---")
    print(f"Total learning sessions: {len(df_learning)}")
    print(f"Average performance: {df_learning['performance'].mean():.2f}")
    print(f"Average time per session: {df_learning['time_spent'].mean():.0f} seconds")
    
    style_performance = df_learning.groupby('learning_style')['performance'].mean()
    print("\nPerformance by learning style:")
    for style, perf in style_performance.items():
        print(f"  {style}: {perf:.2f}")
    
    return df_learning

if __name__ == "__main__":
    learning_data = simulate_learning_journey(20, 45)