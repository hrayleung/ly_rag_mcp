"""
Reranking functionality using Cohere.
"""

from typing import List, Optional
import threading

from rag.config import settings, logger, get_cohere_key
from rag.models import CacheStats


class RerankerManager:
    """
    Manages Cohere reranker with caching.
    """

    def __init__(self):
        self._reranker = None
        self._stats = CacheStats()
        self._lock = threading.RLock()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def reset_stats(self, category: Optional[str] = None) -> None:
        """Reset cache statistics."""
        self._stats.reset(category)

    @property
    def available(self) -> bool:
        """Check if reranker is available."""
        return get_cohere_key() is not None

    def get_reranker(self):
        """
        Get or create Cohere reranker.

        Returns:
            CohereRerank instance or None if not configured
        """
        with self._lock:
            if self._reranker is not None:
                self._stats.reranker_cache_hits +=1
                logger.debug("Reranker cache hit")
                return self._reranker

            self._stats.reranker_loads += 1

            cohere_key = get_cohere_key()
            if not cohere_key:
                logger.debug("Cohere API key not set, reranking disabled")
                return None

            from llama_index.postprocessor.cohere_rerank import CohereRerank

            self._reranker = CohereRerank(
                api_key=cohere_key,
                model="rerank-v3.5",
                top_n=settings.max_top_k
            )

            return self._reranker

    def rerank(
        self,
        nodes: List,
        query: str,
        top_n: int = None
    ) -> List:
        """
        Rerank nodes using Cohere.

        Args:
            nodes: Nodes to rerank
            query: Original query string
            top_n: Number of top results to return

        Returns:
            Reranked nodes
        """
        reranker = self.get_reranker()
        if reranker is None or not nodes:
            return nodes

        try:
            result = reranker.postprocess_nodes(nodes, query_str=query)
            if top_n:
                return result[:min(top_n, len(result))]
            return result
        except Exception as e:
            logger.warning(f"Reranking failed, using original results: {e}")
            # Update stats to reflect the failed rerank attempt
            with self._lock:
                self._stats.reranker_loads += 1
            return nodes[:top_n] if top_n else nodes

    def should_apply_rerank(
        self,
        nodes: List,
        requested: bool = True
    ) -> bool:
        """
        Determine if reranking should be applied.

        Args:
            nodes: Retrieved nodes
            requested: Whether reranking was requested

        Returns:
            True if reranking should be applied
        """
        if not requested or not nodes:
            return False

        if not self.available:
            return False

        if len(nodes) < settings.rerank_min_results:
            return False

        # Check if scores are close enough to warrant reranking
        scores = [n.score for n in nodes if n.score is not None]
        if len(scores) < 2:
            return True

        # Sort scores descending to ensure we get the top two
        sorted_scores = sorted(scores, reverse=True)
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]

        if top_score is None or second_score is None:
            return True

        # If top results are very close, reranking can help
        if abs(top_score - second_score) < settings.rerank_delta_threshold:
            return True

        logger.debug("Skipping rerank (clear winner among candidates)")
        return False


# Global singleton with thread-safe initialization
_reranker_manager: Optional[RerankerManager] = None
_reranker_manager_lock = threading.Lock()


def get_reranker_manager() -> RerankerManager:
    """Get the global RerankerManager instance."""
    global _reranker_manager
    if _reranker_manager is None:
        with _reranker_manager_lock:
            if _reranker_manager is None:
                _reranker_manager = RerankerManager()
    return _reranker_manager
