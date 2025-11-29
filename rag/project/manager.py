"""
Project management functionality.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rag.config import settings, logger
from rag.models import ProjectMetadata
from rag.project.metadata import get_metadata_manager
from rag.storage.index import get_index_manager
from rag.storage.chroma import get_chroma_manager


class ProjectManager:
    """
    Manages project lifecycle and selection.
    """
    
    def __init__(self):
        self._metadata = get_metadata_manager()
        self._current_project = settings.default_project
    
    @property
    def current_project(self) -> str:
        """Get current active project."""
        return self._current_project
    
    def discover_projects(self) -> List[str]:
        """
        Discover available projects in storage.
        
        Returns:
            Sorted list of project names
        """
        if not settings.storage_path.exists():
            return []
        
        projects = []
        for item in settings.storage_path.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith('.'):
                continue
            if item.name == "chroma_db":
                continue
            projects.append(item.name)
        
        return sorted(projects)
    
    def project_exists(self, name: str) -> bool:
        """Check if project exists."""
        return name in self.discover_projects()
    
    def validate_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate project name.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name or not name.strip():
            return False, "Project name cannot be empty"
        
        name = name.strip()
        
        if any(not (c.isalnum() or c in ('_', '-')) for c in name):
            return False, "Project names may only contain letters, numbers, '-' and '_'"
        
        return True, ""
    
    def create_project(self, name: str) -> Dict:
        """
        Create a new project.
        
        Args:
            name: Project name
            
        Returns:
            Result dict with success/error
        """
        # Sanitize name
        safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        
        is_valid, error = self.validate_name(safe_name)
        if not is_valid:
            return {"error": error}
        
        # Check if exists
        project_path = settings.storage_path / safe_name
        if project_path.exists():
            return {"error": f"Project '{safe_name}' already exists"}
        
        logger.info(f"Creating project: {safe_name}")
        
        # Initialize storage
        get_chroma_manager().get_client(safe_name)
        self._metadata.ensure_exists(safe_name)
        
        return {
            "success": True,
            "message": f"Created project '{safe_name}'. Use switch_project('{safe_name}') to activate.",
            "project": safe_name
        }
    
    def switch_project(self, name: str) -> Dict:
        """
        Switch to a different project.
        
        Args:
            name: Project name
            
        Returns:
            Result dict
        """
        if not self.project_exists(name):
            return {
                "error": f"Project '{name}' not found",
                "available": self.discover_projects()
            }
        
        logger.info(f"Switching to project: {name}")
        
        self._current_project = name
        
        # Reset and reload managers
        index_manager = get_index_manager()
        index_manager.switch_project(name)
        
        return {
            "success": True,
            "message": f"Switched to project '{name}'",
            "project": name
        }
    
    def list_projects(self) -> Dict:
        """
        List all projects with details.
        
        Returns:
            Dict with project list and details
        """
        projects = self.discover_projects()
        
        summaries = []
        for project in projects:
            metadata = self._metadata.load(project)
            summaries.append({
                "name": project,
                "display_name": metadata.display_name or project,
                "description": metadata.description or "",
                "keywords": metadata.keywords or [],
                "default_paths": metadata.default_paths or [],
                "last_indexed": metadata.last_indexed,
            })
        
        return {
            "projects": projects,
            "details": summaries,
            "current_project": self._current_project,
            "count": len(projects)
        }
    
    def require_project(
        self,
        action: str,
        project: Optional[str],
        suggested: Optional[str] = None
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Ensure an action has an explicit project target.
        
        Args:
            action: Description of the action
            project: Provided project name
            suggested: Suggested project name
            
        Returns:
            Tuple of (error_dict_or_none, resolved_project_name)
        """
        available = self.discover_projects()
        
        if not project:
            return (
                {
                    "error": "project_confirmation_required",
                    "message": f"{action} requires an explicit project. Ask the user to confirm.",
                    "current_project": self._current_project,
                    "available_projects": available,
                    "suggested_project": suggested,
                    "next_steps": [
                        "create_project('<name>') if new workspace",
                        "switch_project('<name>') if existing",
                        f"Retry: {action}(..., project='<name>')"
                    ]
                },
                None
            )
        
        is_valid, error = self.validate_name(project)
        if not is_valid:
            return ({"error": "invalid_project_name", "message": error}, None)
        
        if project not in available:
            return (
                {
                    "error": "project_not_found",
                    "message": f"Project '{project}' not found. Create it first.",
                    "available_projects": available
                },
                None
            )
        
        # Switch if needed
        if project != self._current_project:
            get_index_manager().get_index(project)
            self._current_project = project
        
        return None, project
    
    def infer_project(self, text: str) -> Optional[str]:
        """
        Infer best matching project from text.
        
        Args:
            text: Query or description text
            
        Returns:
            Project name or None
        """
        projects = self.discover_projects()
        if not projects:
            return None
        
        text_lower = text.lower()
        best_project = None
        best_score = 0
        
        for project in projects:
            metadata = self._metadata.load(project)
            score = self._score_project_match(project, metadata, text_lower)
            
            if score > best_score:
                best_project = project
                best_score = score
        
        return best_project
    
    def _score_project_match(
        self, 
        project: str, 
        metadata: ProjectMetadata, 
        text: str
    ) -> float:
        """Score how well a project matches text."""
        score = 0.0
        project_lower = project.lower()
        
        # Project name match
        if re.search(rf'\b{re.escape(project_lower)}\b', text):
            score += len(project_lower)
        
        # Display name match
        if metadata.display_name:
            name_lower = metadata.display_name.lower()
            if name_lower != project_lower and name_lower in text:
                score += 2
        
        # Keyword matches
        for keyword in (metadata.keywords or []):
            if keyword and keyword.lower() in text:
                score += 3
        
        # Path matches
        for path_hint in (metadata.default_paths or []):
            try:
                tail = Path(path_hint).name.lower()
                if tail and tail in text:
                    score += 1
            except Exception:
                pass
        
        return score
    
    def choose_project(
        self, 
        question: str, 
        max_candidates: int = 3
    ) -> Dict:
        """
        Score and rank projects for a query.
        
        Args:
            question: User query
            max_candidates: Maximum candidates to return
            
        Returns:
            Dict with recommendations
        """
        if not question or not question.strip():
            return {"error": "question_required"}
        
        projects = self.discover_projects()
        if not projects:
            return {"error": "no_projects", "message": "No projects available"}
        
        max_candidates = max(1, min(int(max_candidates), len(projects)))
        question_lower = question.lower()
        
        # Parse directives
        includes, excludes = self._parse_directives(question, projects)
        inferred = self.infer_project(question)
        
        candidates = []
        for project in projects:
            if includes and project not in includes:
                continue
            if project in excludes:
                continue
            
            metadata = self._metadata.load(project)
            score = self._score_project_match(project, metadata, question_lower)
            
            reasons = []
            if project == inferred:
                score += 5
                reasons.append("name/keyword match")
            if project == self._current_project:
                score += 0.5
                reasons.append("current workspace")
            if metadata.last_indexed:
                score += 0.25
            if not metadata.keywords:
                reasons.append("add keywords for better routing")
            
            candidates.append({
                "project": project,
                "score": round(score, 3),
                "reasons": reasons,
                "metadata": metadata.to_dict()
            })
        
        if not candidates:
            return {"error": "no_candidates", "message": "All projects filtered out"}
        
        sorted_candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
        recommendation = sorted_candidates[0] if sorted_candidates else None
        
        return {
            "question": question,
            "current_project": self._current_project,
            "recommendation": recommendation["project"] if recommendation else None,
            "candidates": sorted_candidates[:max_candidates],
            "includes": list(includes),
            "excludes": list(excludes)
        }
    
    def _parse_directives(
        self, 
        question: str, 
        projects: List[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Parse project inclusion/exclusion from query."""
        normalized = question.lower()
        normalized = normalized.replace("'", " ").replace('"', " ")
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        
        includes: Set[str] = set()
        excludes: Set[str] = set()
        
        for project in projects:
            pl = project.lower()
            if pl not in normalized:
                continue
            
            # Check exclusion patterns
            if re.search(rf"(ignore|excluding|except|without|skip)\s.*\b{re.escape(pl)}\b", normalized):
                excludes.add(project)
                continue
            
            # Check inclusion patterns
            if re.search(rf"(only|just|specifically|focus)\s+(the\s+)?{re.escape(pl)}", normalized):
                includes.add(project)
        
        includes -= excludes
        return includes, excludes


# Global singleton
_project_manager: Optional[ProjectManager] = None


def get_project_manager() -> ProjectManager:
    """Get the global ProjectManager instance."""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager
