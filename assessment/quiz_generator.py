import random
from typing import Dict, List, Any, Tuple

class QuizGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.question_bank = {}
        
    def add_questions(self, concept: str, questions: List[Dict[str, Any]]):
        if concept not in self.question_bank:
            self.question_bank[concept] = []
        self.question_bank[concept].extend(questions)
    
    def generate_adaptive_quiz(self, student_profile: Dict[str, Any],
                             target_concepts: List[str],
                             num_questions: int = 10) -> List[Dict[str, Any]]:
        knowledge_state = student_profile.get('knowledge_state', {})
        learning_style = student_profile.get('learning_style', 'reading_writing')
        
        quiz_questions = []
        concept_weights = self.calculate_concept_weights(target_concepts, knowledge_state)
        
        for concept in target_concepts:
            weight = concept_weights.get(concept, 1.0)
            num_concept_questions = max(1, int(num_questions * weight))
            
            if concept in self.question_bank:
                available_questions = self.question_bank[concept]
                selected_questions = self.select_questions_by_style(
                    available_questions, learning_style, num_concept_questions
                )
                quiz_questions.extend(selected_questions)
        
        if len(quiz_questions) > num_questions:
            quiz_questions = random.sample(quiz_questions, num_questions)
        
        random.shuffle(quiz_questions)
        return quiz_questions
    
    def calculate_concept_weights(self, concepts: List[str], knowledge_state: Dict[str, float]) -> Dict[str, float]:
        weights = {}
        for concept in concepts:
            knowledge_level = knowledge_state.get(concept, 0)
            weights[concept] = 1.0 - knowledge_level
        return weights
    
    def select_questions_by_style(self, questions: List[Dict[str, Any]],
                                learning_style: str,
                                num_questions: int) -> List[Dict[str, Any]]:
        style_preferences = {
            'visual': ['diagram', 'image_based'],
            'auditory': ['audio_question', 'listening'],
            'kinesthetic': ['interactive', 'simulation'],
            'reading_writing': ['text', 'multiple_choice']
        }
        
        preferred_types = style_preferences.get(learning_style, ['multiple_choice'])
        
        preferred_questions = [q for q in questions if q.get('type') in preferred_types]
        other_questions = [q for q in questions if q.get('type') not in preferred_types]
        
        selected = preferred_questions[:num_questions]
        if len(selected) < num_questions:
            selected.extend(other_questions[:num_questions - len(selected)])
        
        return selected
    
    def evaluate_quiz_performance(self, quiz_questions: List[Dict[str, Any]],
                                student_answers: Dict[int, Any]) -> Dict[str, Any]:
        correct_count = 0
        concept_performance = {}
        total_questions = len(quiz_questions)
        
        for i, question in enumerate(quiz_questions):
            correct_answer = question.get('correct_answer')
            student_answer = student_answers.get(i)
            
            is_correct = student_answer == correct_answer
            if is_correct:
                correct_count += 1
            
            for concept in question.get('concepts', []):
                if concept not in concept_performance:
                    concept_performance[concept] = {'correct': 0, 'total': 0}
                concept_performance[concept]['total'] += 1
                if is_correct:
                    concept_performance[concept]['correct'] += 1
        
        overall_score = correct_count / total_questions if total_questions > 0 else 0
        
        concept_scores = {}
        for concept, stats in concept_performance.items():
            concept_scores[concept] = stats['correct'] / stats['total']
        
        return {
            'overall_score': overall_score,
            'concept_scores': concept_scores,
            'correct_answers': correct_count,
            'total_questions': total_questions,
            'weak_concepts': [c for c, s in concept_scores.items() if s < 0.6]
        }