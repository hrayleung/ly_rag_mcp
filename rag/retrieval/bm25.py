"""
BM25 retrieval functionality.
"""

from typing import List, Optional
import threading
import time
import hashlib

from rag.config import settings, logger
from rag.models import CacheStats

try:
    from llama_index.retrievers.bm25 import BM25Retriever
except Exception:
    BM25Retriever = None


class BM25Manager:
    """
    Manages BM25 retriever with caching.
    """

    def __init__(self):
        self._retrievers: dict[str, Optional[object]] = {}
        self._doc_count_at_build: dict[str, int] = {}
        self._content_hash_at_build: dict[str, str] = {}
        self._stats = CacheStats()
        self._lock = threading.Lock()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def reset_stats(self, category: Optional[str] = None) -> None:
        """Reset cache statistics."""
        self._stats.reset(category)

    def _compute_content_hash(self, nodes: List) -> str:
        """
        Compute a hash of node content to detect changes.

        Args:
            nodes: List of nodes to hash

        Returns:
            MD5 hash string
        """
        if not nodes:
            return ""

        # Combine text and metadata for each node
        content_parts = []
        for node in sorted(nodes, key=lambda n: getattr(n, 'id_', '')):
            text = getattr(node, 'text', '')
            content_parts.append(text)

            # Include file_path and mtime if available
            metadata = getattr(node, 'metadata', {}) or {}
            file_path = metadata.get('file_path', '')
            mtime_ns = metadata.get('mtime_ns', 0)
            if file_path:
                content_parts.append(f"{file_path}:{mtime_ns}")

        # Compute hash
        content_str = "\n".join(content_parts)
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
    
    def get_retriever(self, index, chroma_collection=None, project: Optional[str] = None):
        """
        Get or create BM25 retriever from index.
        
        Args:
            index: LlamaIndex instance
            chroma_collection: Optional ChromaDB collection for fallback
            
        Returns:
            BM25Retriever instance or None
        """
        cache_key = project or "default"

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

        # Compute content hash for cache validation
        current_hash = self._compute_content_hash(nodes)

        # Ensure cache structures are dictionaries (for backward compatibility with tests)
        if not isinstance(self._doc_count_at_build, dict):
            self._doc_count_at_build = {}
        if not isinstance(self._content_hash_at_build, dict):
            self._content_hash_at_build = {}

        with self._lock:
            cached_retriever = self._retrievers.get(cache_key)
            cached_count = self._doc_count_at_build.get(cache_key, -1)
            cached_hash = self._content_hash_at_build.get(cache_key, "")

            # Check cache with both count and content hash
            if (cached_retriever is not None
                and cached_count == current_count
                and cached_hash == current_hash):
                self._stats.bm25_cache_hits += 1
                logger.debug("BM25 cache hit (content unchanged)")
                return cached_retriever

            # Log when cache is invalidated due to content change
            if (cached_retriever is not None
                and cached_count == current_count
                and cached_hash != current_hash):
                logger.info("BM25 cache invalidated: content changed but count same")
            
            self._stats.bm25_builds += 1
            logger.info(f"Building BM25 retriever with {current_count} nodes")
            
            start_time = time.time()
            retriever_cls = BM25Retriever
            if retriever_cls is None:
                from llama_index.retrievers.bm25 import BM25Retriever as _BM25Retriever
                retriever_cls = _BM25Retriever

            retriever = retriever_cls.from_defaults(
                nodes=nodes,
                similarity_top_k=settings.default_top_k,
                stemmer=None,
                language="english"
            )
            self._retrievers[cache_key] = retriever
            self._doc_count_at_build[cache_key] = current_count
            self._content_hash_at_build[cache_key] = current_hash
            build_time = time.time() - start_time

        logger.info(f"Built BM25 in {build_time:.3f}s")
        return retriever
    
    def _load_nodes_from_chroma(self, collection) -> List:
        """Load nodes from ChromaDB as fallback with pagination."""
        try:
            from llama_index.core.schema import TextNode

            all_nodes = []
            batch_size = 1000  # Process in batches to avoid memory issues
            offset = 0

            while True:
                # Get batch with offset and limit
                result = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=['documents', 'metadatas', 'ids']
                )

                if not result or not result.get('documents'):
                    break

                docs = result.get('documents') or []
                ids = result.get('ids') or []
                metadatas = result.get('metadatas') or []

                limit = min(len(docs), len(ids))
                if limit == 0:
                    break

                for i in range(limit):
                    text = docs[i]
                    if text:
                        node = TextNode(
                            text=text,
                            id_=ids[i],
                            metadata=metadatas[i] if i < len(metadatas) else {}
                        )
                        all_nodes.append(node)

                if len(docs) < batch_size:
                    break

                offset += batch_size

                # Prevent infinite loops
                if offset > 100000:  # Safety limit
                    logger.warning("ChromaDB pagination limit reached, truncating results")
                    break

            logger.info(f"Recovered {len(all_nodes)} nodes from ChromaDB (with pagination)")
            return all_nodes

        except Exception as e:
            logger.error(f"Failed to load from ChromaDB: {e}")
            return []
    
    def reset(self) -> None:
        """Reset the cached retriever."""
        with self._lock:
            self._retrievers = {}
            self._doc_count_at_build = {}
            self._content_hash_at_build = {}


# Global singleton with thread-safe initialization
_bm25_manager: Optional[BM25Manager] = None
_bm25_manager_lock = threading.Lock()


def get_bm25_manager() -> BM25Manager:
    """Get the global BM25Manager instance."""
    global _bm25_manager
    if _bm25_manager is None:
        with _bm25_manager_lock:
            if _bm25_manager is None:
                _bm25_manager = BM25Manager()
    return _bm25_manager
