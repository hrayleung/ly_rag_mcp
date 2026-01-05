"""
Document loading functionality.
"""

import os
from pathlib import Path
from typing import List, Optional, Set, Tuple
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, Document

from rag.config import (
    settings, logger,
    ALLOWED_EXTENSIONS, DEFAULT_EXCLUDES,
    SUPPORTED_CODE_EXTS, SUPPORTED_DOC_EXTS
)


class DocumentLoader:
    """
    Handles loading documents from various sources.
    """

    def validate_file(self, path: Path) -> Tuple[bool, str]:
        """
        Check if a file is eligible for ingestion.

        Args:
            path: Path to file

        Returns:
            Tuple of (is_valid, reason_if_invalid)

        Security:
            - Checks file size BEFORE attempting to read (Bug L7)
            - Uses os.path.getsize() for faster rejection of oversized files
        """
        ext = path.suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            return False, "unsupported_extension"

        # Bug L7: Check file size BEFORE reading (fail-fast)
        try:
            file_size = os.path.getsize(path)
            if file_size > settings.max_file_size_bytes:
                return False, "file_too_large"
        except OSError as e:
            logger.warning(f"Unable to get size of {path}: {e}")
            return False, "stat_error"

        return True, ""

    def scan_directory(
        self,
        directory: Path,
        exclude_patterns: Optional[Set[str]] = None,
        extension_filter: Optional[Set[str]] = None,
        include_hidden: bool = False
    ) -> Tuple[List[Path], dict]:
        """
        Scan directory for files to ingest.

        Args:
            directory: Directory to scan
            exclude_patterns: Patterns to exclude
            extension_filter: Only include these extensions
            include_hidden: Include hidden files

        Returns:
            Tuple of (files_to_process, skip_stats)
        """
        all_excludes = set(DEFAULT_EXCLUDES)
        if exclude_patterns:
            all_excludes.update(exclude_patterns)

        valid_exts = extension_filter or ALLOWED_EXTENSIONS

        files_to_process = []
        skip_stats = {
            "unsupported": [],
            "oversize": [],
            "excluded": [],
            "other": []
        }

        for item in directory.rglob('*'):
            # Skip symlinks to avoid infinite loops
            if item.is_symlink():
                continue

            # Skip hidden files if requested
            if not include_hidden and any(p.startswith('.') for p in item.parts):
                continue

            if not item.is_file():
                continue

            # Check exclusion patterns - use exact match or directory prefix
            relative_path = item.relative_to(directory)
            excluded = False

            for pattern in all_excludes:
                if str(relative_path) == pattern or str(relative_path).startswith(pattern + '/'):
                    excluded = True
                    break

            if excluded:
                skip_stats["excluded"].append(str(item))
                continue

            # Validate file
            is_valid, reason = self.validate_file(item)

            if not is_valid:
                if reason == "unsupported_extension":
                    skip_stats["unsupported"].append(str(item))
                elif reason == "file_too_large":
                    skip_stats["oversize"].append(str(item))
                else:
                    skip_stats["other"].append(str(item))
                continue

            # Check extension filter (already validated, skip redundant check)
            # Extension already validated by validate_file()
            files_to_process.append(item)

        return files_to_process, skip_stats

    def load_files(
        self,
        files: List[Path],
        source_directory: Optional[Path] = None,
        extra_metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Load documents from files.

        Args:
            files: List of file paths
            source_directory: Root directory for relative paths
            extra_metadata: Additional metadata to add

        Returns:
            List of Document objects
        """
        if not files:
            return []

        documents = SimpleDirectoryReader(
            input_files=[str(f) for f in files],
            errors='ignore'
        ).load_data()

        # Log failed files
        if len(documents) < len(files):
            loaded_paths = set()
            for doc in documents:
                path = doc.metadata.get('file_path')
                if path:
                    try:
                        loaded_paths.add(str(Path(path).resolve()))
                    except Exception:
                        pass
            
            for f in files:
                try:
                    if str(f.resolve()) not in loaded_paths:
                        logger.warning(f"Failed to load file: {f}")
                except Exception:
                    pass

        # Add metadata
        now = datetime.now().isoformat()
        for doc in documents:
            doc.metadata['ingested_at'] = now

            # Add file modification time for cache invalidation
            file_path = doc.metadata.get('file_path', '')
            if file_path:
                try:
                    stat = Path(file_path).stat()
                    # Bug M6 fix: Use fallback for Python < 3.3 or platforms without st_mtime_ns
                    try:
                        doc.metadata['mtime_ns'] = stat.st_mtime_ns
                    except AttributeError:
                        # Fallback for older Python versions or platforms
                        doc.metadata['mtime_ns'] = int(stat.st_mtime * 1e9)
                except Exception as e:
                    logger.debug(f"Failed to get mtime for {file_path}: {e}")

            if source_directory:
                doc.metadata['source_directory'] = str(source_directory)

            # Classify content type
            file_path = doc.metadata.get('file_path', '')
            ext = Path(file_path).suffix.lower()

            if ext in SUPPORTED_CODE_EXTS:
                doc.metadata['content_type'] = 'code'
            elif ext in SUPPORTED_DOC_EXTS:
                doc.metadata['content_type'] = 'document'

            # Add extra metadata
            if extra_metadata:
                doc.metadata.update(extra_metadata)

        return documents

    def load_directory(
        self,
        directory: Path,
        exclude_patterns: Optional[Set[str]] = None,
        extension_filter: Optional[Set[str]] = None,
        include_hidden: bool = False,
        extra_metadata: Optional[dict] = None
    ) -> Tuple[List[Document], dict]:
        """
        Load all documents from a directory.

        Args:
            directory: Directory to load from
            exclude_patterns: Patterns to exclude
            extension_filter: Only include these extensions
            include_hidden: Include hidden files
            extra_metadata: Additional metadata

        Returns:
            Tuple of (documents, skip_stats)
        """
        files, skip_stats = self.scan_directory(
            directory=directory,
            exclude_patterns=exclude_patterns,
            extension_filter=extension_filter,
            include_hidden=include_hidden
        )

        documents = self.load_files(
            files=files,
            source_directory=directory,
            extra_metadata=extra_metadata
        )

        return documents, skip_stats
