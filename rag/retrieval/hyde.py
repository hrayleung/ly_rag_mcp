"""
HyDE (Hypothetical Document Embedding) functionality.
"""

import os
from typing import List

from rag.config import settings, logger


def generate_hyde_query(question: str) -> str:
    """
    Generate a hypothetical answer for the question using LLM.
    
    HyDE creates a synthetic document that might answer the question,
    which can improve retrieval for ambiguous queries.
    
    Args:
        question: Original question
        
    Returns:
        Generated hypothetical document, or original question on failure
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY missing, skipping HyDE")
        return question
    
    try:
        from llama_index.llms.openai import OpenAI
        
        llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_key)
        prompt = (
            "Please write a brief passage to answer the question.\n"
            "Question: {question}\n"
            "Passage:"
        )
        response = llm.complete(prompt.format(question=question))
        
        result = str(response)
        logger.debug(f"HyDE generated: {result[:100]}...")
        return result
        
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return question


def should_trigger_hyde(nodes: List) -> bool:
    """
    Determine whether to fall back to HyDE augmentation.
    
    Args:
        nodes: Retrieved nodes
        
    Returns:
        True if HyDE should be triggered
    """
    if not nodes:
        return True
    
    scores = [n.score for n in nodes if n.score is not None]
    if not scores:
        return True
    
    # Trigger if very few results with low scores
    if (len(scores) <= settings.hyde_trigger_min_results 
        and scores[0] < settings.hyde_trigger_score):
        return True
    
    # Trigger if all scores are below threshold
    if max(scores) < settings.low_score_threshold:
        return True
    
    return False
