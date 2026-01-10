"""
Data models for RAG system.

Defines structured data types used throughout application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class SearchMode(str, Enum):
    """Available search modes."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class ChangeType(str, Enum):
    """File change types for incremental indexing."""
    NEW = "new"
    MODIFIED = "modified"
    REMOVED = "removed"


class ContentType(str, Enum):
    """Content type classification."""
    CODE = "code"
    DOCUMENT = "document"
    MIXED = "mixed"


@dataclass
class FileMetadata:
    """Metadata for tracked files."""
    path: str
    mtime_ns: int
    size: int = 0

    @classmethod
    def from_path(cls, path) -> "FileMetadata":
        """Create FileMetadata from a Path object."""
        from pathlib import Path
        p = Path(path)
        stat = p.stat()
        return cls(
            path=str(p),
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size
        )


@dataclass
class ProjectMetadata:
    """Project configuration and metadata."""
    name: str
    display_name: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    default_paths: List[str] = field(default_factory=list)
    last_indexed: Optional[str] = None
    last_chat_time: Optional[str] = None
    chat_turn_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self):
        # Only auto-set created_at if it's truly None, not just falsy
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        # Only auto-set updated_at if it's truly None (preserves loaded values)
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()

        # Default display_name to name if not set
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "keywords": self.keywords,
            "default_paths": self.default_paths,
            "last_indexed": self.last_indexed,
            "last_chat_time": self.last_chat_time,
            "chat_turn_count": self.chat_turn_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectMetadata":
        """Create from dictionary."""
        name = data.get("name")
        if not name:
            raise ValueError("ProjectMetadata requires a non-empty 'name' field")
        return cls(
            name=name,
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            keywords=data.get("keywords", []),
            default_paths=data.get("default_paths", []),
            last_indexed=data.get("last_indexed"),
            last_chat_time=data.get("last_chat_time"),
            chat_turn_count=data.get("chat_turn_count", 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class RetrievalResult:
    """
    Result from a retrieval operation.

    Attributes:
        text: The retrieved text content
        score: Relevance score (0.0-1.0) or None. None indicates the score is
               unavailable or not applicable (e.g., for certain retriever types
               or during error conditions). Code should handle both float and None.
        metadata: Additional metadata about the result
        node_id: Unique identifier for the node in the index
    """
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = ""

    @property
    def preview(self) -> str:
        """Get a preview of the text."""
        if len(self.text) <= 200:
            return self.text
        # Try to break at a space for cleaner preview
        truncated = self.text[:200]
        last_space = truncated.rfind(' ')
        if last_space > 150:
            return truncated[:last_space] + "..."
        return truncated + "..."


@dataclass
class SearchResult:
    """Complete search result with metadata."""
    results: List[RetrievalResult]
    query: str
    search_mode: SearchMode
    reranked: bool = False
    used_hyde: bool = False
    generated_query: Optional[str] = None
    project: str = ""

    @property
    def total(self) -> int:
        return len(self.results)


@dataclass
class IngestResult:
    """Result from an ingestion operation."""
    success: bool
    message: str
    documents_processed: int = 0
    chunks_created: int = 0
    skipped_unsupported: int = 0
    skipped_oversize: int = 0
    skipped_other: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "success": self.success,
            "message": self.message,
            "documents_processed": self.documents_processed,
            "chunks_created": self.chunks_created,
        }
        if self.skipped_unsupported:
            result["skipped_unsupported"] = self.skipped_unsupported
        if self.skipped_oversize:
            result["skipped_oversize"] = self.skipped_oversize
        if self.skipped_other:
            result["skipped_other"] = self.skipped_other
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class CacheStats:
    """Cache performance statistics."""
    index_loads: int = 0
    index_cache_hits: int = 0
    reranker_loads: int = 0
    reranker_cache_hits: int = 0
    chroma_loads: int = 0
    chroma_cache_hits: int = 0
    bm25_builds: int = 0
    bm25_cache_hits: int = 0

    VALID_CATEGORIES = {"index", "reranker", "chroma", "bm25"}

    def get_hit_rate(self, category: str) -> float:
        """Calculate hit rate for a category."""
        if category not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Valid categories: {self.VALID_CATEGORIES}")

        # Map category to (loads_attr, hits_attr)
        attr_map = {
            "index": ("index_loads", "index_cache_hits"),
            "reranker": ("reranker_loads", "reranker_cache_hits"),
            "chroma": ("chroma_loads", "chroma_cache_hits"),
            "bm25": ("bm25_builds", "bm25_cache_hits"),
        }
        loads_attr, hits_attr = attr_map[category]
        loads = getattr(self, loads_attr)
        hits = getattr(self, hits_attr)
        total = loads + hits
        return (hits / total * 100) if total > 0 else 0

    def reset(self, category: Optional[str] = None) -> None:
        """Reset statistics for a category or all categories."""
        if category:
            if category not in self.VALID_CATEGORIES:
                raise ValueError(f"Invalid category: {category}. Valid categories: {self.VALID_CATEGORIES}")

            # Map category to attribute names to reset
            attr_map = {
                "index": ("index_loads", "index_cache_hits"),
                "reranker": ("reranker_loads", "reranker_cache_hits"),
                "chroma": ("chroma_loads", "chroma_cache_hits"),
                "bm25": ("bm25_builds", "bm25_cache_hits"),
            }
            for attr in attr_map[category]:
                setattr(self, attr, 0)
        else:
            # Reset all
            self.index_loads = 0
            self.index_cache_hits = 0
            self.reranker_loads = 0
            self.reranker_cache_hits = 0
            self.chroma_loads = 0
            self.chroma_cache_hits = 0
            self.bm25_builds = 0
            self.bm25_cache_hits = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with hit rates."""
        return {
            "index": {
                "loads": self.index_loads,
                "cache_hits": self.index_cache_hits,
                "hit_rate": f"{self.get_hit_rate('index'):.1f}%"
            },
            "reranker": {
                "loads": self.reranker_loads,
                "cache_hits": self.reranker_cache_hits,
                "hit_rate": f"{self.get_hit_rate('reranker'):.1f}%"
            },
            "chroma": {
                "loads": self.chroma_loads,
                "cache_hits": self.chroma_cache_hits,
                "hit_rate": f"{self.get_hit_rate('chroma'):.1f}%"
            },
            "bm25": {
                "builds": self.bm25_builds,
                "cache_hits": self.bm25_cache_hits,
                "hit_rate": f"{self.get_hit_rate('bm25'):.1f}%"
            }
        }
