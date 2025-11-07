from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class LearningRequest(BaseModel):
    student_id: str
    target_concepts: List[str]
    max_recommendations: Optional[int] = 5

class LearningResponse(BaseModel):
    student_id: str
    recommendations: List[Dict[str, Any]]
    timestamp: str