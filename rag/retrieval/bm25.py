"""
BM25 retrieval functionality.
"""

from typing import List, Optional
import time

from rag.config import settings, logger
from rag.models import CacheStats


class BM25Manager:
    """
    Manages BM25 retriever with caching.
    """
    
    def __init__(self):
        self._retriever = None
        self._doc_count_at_build: int = 0
        self._stats = CacheStats()
    
    @property
    def stats(self) -> CacheStats:
        return self._stats
    
    def get_retriever(self, index, chroma_collection=None):
        """
        Get or create BM25 retriever from index.
        
        Args:
            index: LlamaIndex instance
            chroma_collection: Optional ChromaDB collection for fallback
            
        Returns:
            BM25Retriever instance or None
        """
        # Get nodes from docstore
        nodes = list(index.docstore.docs.values())
        current_count = len(nodes)
        
        # Fallback to ChromaDB if docstore is empty
        if current_count == 0 and chroma_collection:
            nodes = self._load_nodes_from_chroma(chroma_collection)
            current_count = len(nodes)
        
        if current_count == 0:
            logger.warning("No documents available for BM25")
            return None
        
        # Check cache
        if (self._retriever is not None 
            and self._doc_count_at_build == current_count):
            self._stats.bm25_cache_hits += 1
            logger.debug("BM25 cache hit")
            return self._retriever
        
        self._stats.bm25_builds += 1
        logger.info(f"Building BM25 retriever with {current_count} nodes")
        
        start_time = time.time()
        
        from llama_index.retrievers.bm25 import BM25Retriever
        
        self._retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=settings.default_top_k,
            stemmer=None,
            language="english"
        )
        self._doc_count_at_build = current_count
        
        logger.info(f"Built BM25 in {time.time() - start_time:.3f}s")
        return self._retriever
    
    def _load_nodes_from_chroma(self, collection) -> List:
        """Load nodes from ChromaDB as fallback."""
        try:
            from llama_index.core.schema import TextNode
            
            result = collection.get()
            if not result or not result.get('documents'):
                return []
            
            nodes = []
            for i, text in enumerate(result['documents']):
                if text:
                    node = TextNode(
                        text=text,
                        id_=result['ids'][i],
                        metadata=result['metadatas'][i] if result.get('metadatas') else {}
                    )
                    nodes.append(node)
            
            logger.info(f"Recovered {len(nodes)} nodes from ChromaDB")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to load from ChromaDB: {e}")
            return []
    
    def reset(self) -> None:
        """Reset the cached retriever."""
        self._retriever = None
        self._doc_count_at_build = 0


# Global singleton
_bm25_manager: Optional[BM25Manager] = None


def get_bm25_manager() -> BM25Manager:
    """Get the global BM25Manager instance."""
    global _bm25_manager
    if _bm25_manager is None:
        _bm25_manager = BM25Manager()
    return _bm25_manager
