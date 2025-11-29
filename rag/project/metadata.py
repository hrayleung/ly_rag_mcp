"""
Project metadata management.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.config import settings, logger
from rag.models import ProjectMetadata


class MetadataManager:
    """
    Manages project metadata storage and retrieval.
    """
    
    def __init__(self):
        self._cache: Dict[str, ProjectMetadata] = {}
    
    def _get_metadata_path(self, project: str) -> Path:
        """Get path to metadata file for a project."""
        return settings.storage_path / project / settings.project_metadata_filename
    
    def _get_manifest_path(self, project: str) -> Path:
        """Get path to ingest manifest for a project."""
        return settings.storage_path / project / settings.ingest_manifest_filename
    
    def load(self, project: str) -> ProjectMetadata:
        """
        Load metadata for a project.
        
        Args:
            project: Project name
            
        Returns:
            ProjectMetadata instance
        """
        if project in self._cache:
            return self._cache[project]
        
        meta_path = self._get_metadata_path(project)
        metadata = ProjectMetadata(name=project)
        
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        metadata = ProjectMetadata.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {project}: {e}")
        
        self._cache[project] = metadata
        return metadata
    
    def save(
        self, 
        project: str, 
        metadata: Optional[ProjectMetadata] = None,
        initialize: bool = False
    ) -> ProjectMetadata:
        """
        Save metadata for a project.
        
        Args:
            project: Project name
            metadata: Metadata to save (creates default if None)
            initialize: If True and no description, set empty string
            
        Returns:
            Saved metadata
        """
        meta_path = self._get_metadata_path(project)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        if metadata is None:
            metadata = ProjectMetadata(name=project)
        
        # Update timestamps
        metadata.updated_at = datetime.now().isoformat()
        if not metadata.created_at:
            metadata.created_at = metadata.updated_at
        
        if initialize and not metadata.description:
            metadata.description = ""
        
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, sort_keys=True)
        
        self._cache[project] = metadata
        return metadata
    
    def ensure_exists(self, project: str) -> ProjectMetadata:
        """Ensure metadata file exists for project."""
        meta_path = self._get_metadata_path(project)
        if not meta_path.exists():
            return self.save(project, initialize=True)
        return self.load(project)
    
    def update(
        self,
        project: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        default_paths: Optional[List[str]] = None
    ) -> ProjectMetadata:
        """
        Update specific metadata fields.
        
        Args:
            project: Project name
            display_name: New display name
            description: New description
            keywords: New keywords list
            default_paths: New default paths
            
        Returns:
            Updated metadata
        """
        metadata = self.load(project)
        
        if display_name is not None:
            metadata.display_name = display_name.strip() or project
        
        if description is not None:
            metadata.description = description.strip()
        
        if keywords is not None:
            metadata.keywords = self._normalize_list(keywords)
        
        if default_paths is not None:
            metadata.default_paths = self._normalize_list(default_paths)
        
        return self.save(project, metadata)
    
    def record_index_activity(
        self, 
        project: str, 
        source_hint: Optional[str] = None
    ) -> None:
        """Record indexing activity in metadata."""
        metadata = self.load(project)
        metadata.last_indexed = datetime.now().isoformat()
        
        if source_hint:
            paths = metadata.default_paths or []
            if source_hint not in paths:
                paths = (paths + [source_hint])[-5:]  # Keep last 5
                metadata.default_paths = paths
        
        self.save(project, metadata)
    
    def _normalize_list(self, value: Any) -> List[str]:
        """Normalize various inputs to list of strings."""
        if value is None:
            return []
        
        if isinstance(value, str):
            items = re.split(r"[,\n]", value)
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]
        
        return [str(item).strip() for item in items if str(item).strip()]
    
    # Manifest methods for incremental indexing
    
    def load_manifest(self, project: str) -> dict:
        """Load ingest manifest for incremental tracking."""
        manifest_path = self._get_manifest_path(project)
        
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data.setdefault("roots", {})
                        return data
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        
        return {"roots": {}, "updated_at": datetime.now().isoformat()}
    
    def save_manifest(self, project: str, manifest: dict) -> None:
        """Save ingest manifest."""
        manifest_path = self._get_manifest_path(project)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest["updated_at"] = datetime.now().isoformat()
        
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    
    def clear_cache(self, project: Optional[str] = None) -> None:
        """Clear metadata cache."""
        if project:
            self._cache.pop(project, None)
        else:
            self._cache.clear()


# Global singleton
_metadata_manager: Optional[MetadataManager] = None


def get_metadata_manager() -> MetadataManager:
    """Get the global MetadataManager instance."""
    global _metadata_manager
    if _metadata_manager is None:
        _metadata_manager = MetadataManager()
    return _metadata_manager
