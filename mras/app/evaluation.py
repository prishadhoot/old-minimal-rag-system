"""
Evaluation Module

Dataset: JSON list of questions + expected keywords

Metrics:
- Keyword match score
- Average latency

Output: JSON metrics summary

Constraints:
- No external RAG evaluation frameworks
- Pure Python
"""

import time
from typing import List, Dict
from app.models import EvalSample
from app.agent import Agent


def evaluate(agent: Agent, eval_samples: List[EvalSample]) -> Dict[str, float]:
    """
    Evaluate agent performance.
    
    Args:
        agent: Agent instance to evaluate
        eval_samples: List of EvalSample objects
        
    Returns:
        Dictionary with:
        - avg_keyword_score: Average fraction of expected keywords present
        - avg_latency: Average response time in seconds
    """
    total_keyword_score = 0.0
    total_latency = 0.0
    
    for sample in eval_samples:
        # Measure latency
        start_time = time.time()
        response = agent.answer(sample.question)
        latency = time.time() - start_time
        
        # Calculate keyword score
        keyword_score = _calculate_keyword_score(
            response.answer,
            sample.expected_keywords
        )
        
        total_keyword_score += keyword_score
        total_latency += latency
    
    num_samples = len(eval_samples)
    
    return {
        "avg_keyword_score": total_keyword_score / num_samples if num_samples > 0 else 0.0,
        "avg_latency": total_latency / num_samples if num_samples > 0 else 0.0
    }


def _calculate_keyword_score(answer: str, expected_keywords: List[str]) -> float:
    """
    Calculate fraction of expected keywords present in answer.
    
    Args:
        answer: Generated answer string
        expected_keywords: List of expected keywords
        
    Returns:
        Fraction of keywords found (0.0 to 1.0)
    """
    if not expected_keywords:
        return 1.0
    
    answer_lower = answer.lower()
    matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    
    return matches / len(expected_keywords)
