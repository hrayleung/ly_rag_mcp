"""
Document processing and cleaning.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from llama_index.core import Document

from rag.config import logger


# Allowed metadata value types for vector stores
_ALLOWED_METADATA_TYPES = (str, int, float, bool, type(None))


class DocumentProcessor:
    """
    Handles document cleaning and enrichment.
    """
    
    def clean_text(self, text: str) -> str:
        """
        Clean document text encoding.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        try:
            # Handle encoding issues
            cleaned = text.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='ignore')
        except Exception:
            cleaned = text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Remove non-printable characters (except whitespace)
        cleaned = ''.join(c for c in cleaned if c.isprintable() or c in '\n\t ')
        
        return cleaned
    
    def sanitize_metadata(self, metadata: Optional[Dict]) -> Dict[str, Any]:
        """
        Ensure metadata only contains values acceptable by vector stores.
        
        Args:
            metadata: Raw metadata dict
            
        Returns:
            Sanitized metadata dict
        """
        if not isinstance(metadata, dict):
            return {}
        
        sanitized = {}
        
        for key, value in metadata.items():
            safe_key = str(key)
            
            if isinstance(value, _ALLOWED_METADATA_TYPES):
                sanitized[safe_key] = value
            elif isinstance(value, (list, tuple, set)):
                sanitized[safe_key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                try:
                    import json
                    sanitized[safe_key] = json.dumps(value, ensure_ascii=False)
                except Exception:
                    sanitized[safe_key] = str(value)
            else:
                sanitized[safe_key] = str(value)
        
        return sanitized
    
    def inject_context(self, doc: Document, root_path: Optional[Path] = None) -> Document:
        """
        Prepend folder context and filename to document text.
        
        This helps RAG understand file structure and locations.
        
        Args:
            doc: Document to process
            root_path: Root path for relative path calculation
            
        Returns:
            Document with context injected
        """
        try:
            # Get source path
            source = (
                doc.metadata.get("file_path")
                or doc.metadata.get("relative_path")
                or doc.metadata.get("source")
            )
            
            if not source:
                return doc
            
            source_str = str(source)
            context_str = ""
            item_name = ""
            
            # Handle URLs
            if source_str.startswith(("http://", "https://")):
                context_str, item_name = self._extract_url_context(source_str)
            else:
                context_str, item_name = self._extract_file_context(
                    source_str, root_path, doc.metadata
                )
            
            # Build prefix
            prefix = ""
            if context_str:
                prefix += f"Context: {context_str}\n"
            prefix += f"Filename: {item_name}\n\n"
            
            # Apply to text
            text = getattr(doc, "text", "")
            if text and not text.startswith("Context: ") and not text.startswith("Filename: "):
                doc.text = prefix + text
            
            # Update metadata
            doc.metadata.setdefault("filename", item_name)
            if context_str:
                doc.metadata["folder_context"] = context_str
            
            return doc
            
        except Exception as e:
            logger.debug(f"Failed to inject context: {e}")
            return doc
    
    def _extract_url_context(self, url: str) -> tuple:
        """Extract context from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")
            
            item_name = path_parts[-1] if path_parts and path_parts[-1] else "index"
            ctx_parts = [parsed.netloc] + path_parts[:-1]
            context_str = " / ".join(ctx_parts)
            
            return context_str, item_name
        except Exception:
            return "", url
    
    def _extract_file_context(
        self, 
        source: str, 
        root_path: Optional[Path],
        metadata: dict
    ) -> tuple:
        """Extract context from file path."""
        path_obj = Path(source)
        item_name = path_obj.name or source
        
        # Try to get root from metadata
        root = root_path or Path(
            metadata.get("source_directory")
            or metadata.get("codebase_root")
            or metadata.get("root_path")
            or ""
        )
        
        context_parts = []
        
        if root and str(root):
            try:
                rel_path = path_obj.relative_to(root)
                context_parts = list(rel_path.parent.parts)
            except ValueError:
                pass
        
        # Fallback: use last 2 parent directories
        if not context_parts:
            parents = [p for p in path_obj.parent.parts if p and p not in ('/', '\\')]
            if len(parents) >= 2:
                context_parts = parents[-2:]
            elif len(parents) == 1:
                context_parts = parents
        
        context_parts = [p for p in context_parts if p != "."]
        context_str = " / ".join(context_parts)
        
        return context_str, item_name
    
    def process_documents(
        self, 
        documents: List[Document],
        root_path: Optional[Path] = None,
        min_content_length: int = 10
    ) -> List[Document]:
        """
        Process and clean a list of documents.
        
        Args:
            documents: Documents to process
            root_path: Root path for context injection
            min_content_length: Minimum text length to keep
            
        Returns:
            List of processed documents
        """
        processed = []
        
        for doc in documents:
            try:
                # Clean text
                content = doc.get_content()
                cleaned = self.clean_text(content)
                
                # Skip empty/tiny documents
                if len(cleaned.strip()) < min_content_length:
                    continue
                
                # Create new document with cleaned content
                new_doc = Document(
                    text=cleaned,
                    metadata=self.sanitize_metadata(doc.metadata),
                    excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
                    excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                )
                
                # Inject context
                new_doc = self.inject_context(new_doc, root_path)
                
                processed.append(new_doc)
                
            except Exception as e:
                logger.warning(f"Skipping problematic document: {e}")
                continue
        
        skipped = len(documents) - len(processed)
        if skipped > 0:
            logger.info(f"Processed {len(processed)} documents ({skipped} skipped)")
        
        return processed
