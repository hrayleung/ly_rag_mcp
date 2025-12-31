"""
HyDE (Hypothetical Document Embedding) functionality.
"""

import os
import time
import threading
from typing import List
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from rag.config import settings, logger


MAX_RETRIES = settings.hyde_max_retries
INITIAL_BACKOFF_SECONDS = settings.hyde_initial_backoff

_hyde_cache_lock = threading.Lock()
_hyde_cache: dict[str, str] = {}


def generate_hyde_query(question: str) -> str:
    """
    Generate a hypothetical answer for the question using LLM.
    Falls back to the original question on failure.
    """
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY missing, skipping HyDE")
        return question

    try:
        return _generate_hyde_query_cached(question)
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return question


def _generate_hyde_query_cached(question: str) -> str:
    openai_key = os.getenv("OPENAI_API_KEY")
    from llama_index.llms.openai import OpenAI

    cache_key = question

    with _hyde_cache_lock:
        if cache_key in _hyde_cache:
            return _hyde_cache[cache_key]

    prompt = (
        "Please write a brief passage to answer the question.\n"
        "Question: {question}\n"
        "Passage:"
    )
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            llm = OpenAI(
                model="gpt-3.5-turbo",
                api_key=openai_key,
                timeout=settings.hyde_timeout,
            )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    llm.complete,
                    prompt.format(question=question)
                )
                response = future.result(timeout=settings.hyde_timeout)

            result = str(response)
            logger.debug(f"HyDE generated: {result[:100]}...")

            with _hyde_cache_lock:
                _hyde_cache[cache_key] = result

            return result
        except TimeoutError:
            logger.error("HyDE generation timed out")
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "HyDE generation attempt %s failed: %s", attempt + 1, e
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


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
