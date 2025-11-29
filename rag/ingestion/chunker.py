"""
Document chunking strategies.
"""

from enum import Enum
from pathlib import Path
from typing import List

from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.core import Document

from rag.config import settings, logger, CODE_LANGUAGE_MAP


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    TEXT = "text"
    CODE = "code"
    AUTO = "auto"


def get_language_for_file(file_path: str) -> str | None:
    """
    Get programming language for a file extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        Language name or None
    """
    ext = Path(file_path).suffix.lower()
    return CODE_LANGUAGE_MAP.get(ext)


def get_splitter_for_file(file_path: str):
    """
    Get the optimal splitter based on file extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        Node parser instance
    """
    language = get_language_for_file(file_path)
    
    if language:
        try:
            return CodeSplitter(
                language=language,
                chunk_lines=settings.code_chunk_lines,
                chunk_lines_overlap=settings.code_chunk_overlap,
                max_chars=settings.code_max_chars,
            )
        except Exception as e:
            logger.warning(f"CodeSplitter failed for {language}: {e}")
    
    # Fallback to text splitter
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


class DocumentChunker:
    """
    Handles document chunking with strategy selection.
    """
    
    def __init__(self, strategy: ChunkingStrategy = ChunkingStrategy.AUTO):
        self.strategy = strategy
        self._text_splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    
    def chunk_document(self, document: Document) -> List:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of nodes
        """
        if self.strategy == ChunkingStrategy.TEXT:
            return self._text_splitter.get_nodes_from_documents([document])
        
        if self.strategy == ChunkingStrategy.CODE:
            file_path = document.metadata.get('file_path', '')
            splitter = get_splitter_for_file(file_path)
            return splitter.get_nodes_from_documents([document])
        
        # AUTO strategy
        file_path = document.metadata.get('file_path', '')
        splitter = get_splitter_for_file(file_path)
        return splitter.get_nodes_from_documents([document])
    
    def chunk_documents(self, documents: List[Document]) -> List:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: Documents to chunk
            
        Returns:
            List of all nodes
        """
        all_nodes = []
        
        for doc in documents:
            try:
                nodes = self.chunk_document(doc)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.warning(f"Failed to chunk document: {e}")
                # Fallback to text splitter
                nodes = self._text_splitter.get_nodes_from_documents([doc])
                all_nodes.extend(nodes)
        
        return all_nodes
