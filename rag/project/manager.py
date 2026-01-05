"""
Project management functionality.
"""

import re
import threading
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
        self._lock = threading.RLock()
    
    @property
    def current_project(self) -> str:
        """Get current active project."""
        with self._lock:
            return self._current_project
    
    def discover_projects(self) -> List[str]:
        """
        Discover available projects in storage.

        Returns:
            Sorted list of project names
        """
        with self._lock:
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

                project_path = item
                has_metadata = (project_path / settings.project_metadata_filename).exists()
                has_chroma = (project_path / "chroma_db").exists()
                has_docstore = (project_path / "docstore.json").exists()
                has_index_store = (project_path / "index_store.json").exists()

                if has_metadata or has_chroma or has_docstore or has_index_store:
                    projects.append(item.name)
                else:
                    logger.debug(
                        "Skipping project candidate without markers: %s", project_path
                    )

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

        # Use lock to prevent TOCTOU race condition
        with self._lock:
            project_path = settings.storage_path / safe_name

            # Check if exists (inside lock now)
            if project_path.exists():
                return {"error": f"Project '{safe_name}' already exists"}

            logger.info(f"Creating project: {safe_name}")

            try:
                # Atomic directory creation
                project_path.mkdir(exist_ok=False)

                # Initialize storage
                get_chroma_manager().get_client(safe_name)
                self._metadata.ensure_exists(safe_name)

            except FileExistsError:
                return {"error": f"Project '{safe_name}' already exists"}
            except Exception as e:
                logger.error(f"Failed to create project: {e}", exc_info=True)
                return {"error": f"Failed to create project: {e}"}

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

        # Acquire lock before changing state
        with self._lock:
            self._current_project = name

        # Bug M10 fix: Clear metadata cache when switching projects
        # This prevents stale metadata after external project switches
        self._metadata.clear_cache()

        # Reset and reload managers (outside lock to avoid deadlock)
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

        # Use lock when reading current_project (Bug M12)
        with self._lock:
            current = self._current_project

        return {
            "projects": projects,
            "details": summaries,
            "current_project": current,
            "count": len(projects)
        }
    
    def set_project_metadata(
        self,
        project: str,
        keywords: List[str] = None,
        description: str = None
    ) -> None:
        """Update project metadata."""
        metadata = self._metadata.load(project)
        if keywords is not None:
            metadata.keywords = keywords
        if description is not None:
            metadata.description = description
        self._metadata.save(project, metadata)
    
    def update_project_paths(self, project: str, paths: List[str]) -> None:
        """Update default paths for a project."""
        metadata = self._metadata.load(project)
        for path in paths:
            if path not in metadata.default_paths:
                metadata.default_paths.append(path)
        self._metadata.save(project, metadata)
    
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

        # Switch if needed - atomic with lock
        with self._lock:
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
            logger.debug("infer_project: No projects discovered, returning None")
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
        
        # Keyword matches - use word boundary for higher score (Bug M5)
        for keyword in (metadata.keywords or []):
            if not keyword:
                continue
            keyword_lower = keyword.lower()
            # Exact word boundary match gets higher score
            if re.search(rf'\b{re.escape(keyword_lower)}\b', text):
                score += 5
            # Partial substring match gets lower score
            elif keyword_lower in text:
                score += 2
        
        # Path matches
        for path_hint in (metadata.default_paths or []):
            try:
                tail = Path(path_hint).name.lower()
                if tail and tail in text:
                    score += 1
            except Exception:
                pass

        # File extension matches (indicates project type)
        common_extensions = {
            '.py': 'python',
            '.js': 'javascript', '.ts': 'typescript', '.jsx': 'react', '.tsx': 'react',
            '.go': 'go', '.rs': 'rust', '.java': 'java', '.cpp': 'cpp', '.c': 'c',
            '.rb': 'ruby', '.php': 'php', '.swift': 'swift', '.kt': 'kotlin',
            '.scala': 'scala', '.r': 'r', '.sh': 'shell', '.sql': 'sql',
            '.vue': 'vue', '.svelte': 'svelte', '.astro': 'astro'
        }

        for ext, lang in common_extensions.items():
            if ext in text.lower() and lang in project_lower:
                score += 2

        # Framework/library mentions
        framework_patterns = {
            'react': ['react', 'jsx', 'hooks', 'components'],
            'vue': ['vue', 'template', 'reactive'],
            'django': ['django', 'models', 'views', 'urls'],
            'flask': ['flask', 'blueprint', 'jinja'],
            'express': ['express', 'middleware', 'routes'],
            'fastapi': ['fastapi', 'pydantic', 'endpoint'],
            'spring': ['spring', 'bean', 'autowired'],
            'rails': ['rails', 'activerecord', 'migration'],
        }

        for framework, keywords in framework_patterns.items():
            if framework in project_lower:
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1.5
                        break

        # Description matching (if available)
        if metadata.description:
            desc_words = set(re.findall(r'\b\w+\b', metadata.description.lower()))
            text_words = set(re.findall(r'\b\w+\b', text))
            overlap = len(desc_words & text_words)
            if overlap > 0:
                score += overlap * 0.5

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

        # Validate and clamp max_candidates (Bug M13)
        try:
            max_candidates_int = int(max_candidates)
            if max_candidates_int < 1:
                return {"error": "invalid_max_candidates", "message": "max_candidates must be >= 1"}
            max_candidates_int = min(max_candidates_int, len(projects))
        except (ValueError, TypeError):
            return {"error": "invalid_max_candidates", "message": "max_candidates must be a valid integer"}

        question_lower = question.lower()

        # Parse directives
        includes, excludes = self._parse_directives(question, projects)
        inferred = self.infer_project(question)

        # Read current_project once to avoid race conditions
        with self._lock:
            current = self._current_project

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
            if project == current:
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
            "current_project": current,
            "recommendation": recommendation["project"] if recommendation else None,
            "candidates": sorted_candidates[:max_candidates_int],
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
        # Keep underscores and dashes in normalized text (Bug M11)
        normalized = re.sub(r"[^a-z0-9\s_-]", " ", normalized)

        includes: Set[str] = set()
        excludes: Set[str] = set()

        for project in projects:
            pl = project.lower()
            if pl not in normalized:
                continue

            # Check exclusion patterns (Bug M11: handle dashes/underscores in project names)
            # Use lookaround for word boundaries that work with dashes/underscores
            exclusion_pattern = rf"(ignore|excluding|except|without|skip).*?(?<!\w){re.escape(pl)}(?!\w)"
            if re.search(exclusion_pattern, normalized):
                excludes.add(project)
                continue

            # Check inclusion patterns (Bug M11: handle dashes/underscores in project names)
            # Use lookaround for word boundaries that work with dashes/underscores
            inclusion_pattern = rf"(only|just|specifically|focus).*?(?<!\w){re.escape(pl)}(?!\w)"
            if re.search(inclusion_pattern, normalized):
                includes.add(project)

        includes -= excludes
        return includes, excludes


# Global singleton with thread-safe initialization
_project_manager: Optional[ProjectManager] = None
_project_manager_lock = threading.Lock()


def get_project_manager() -> ProjectManager:
    """Get the global ProjectManager instance."""
    global _project_manager
    if _project_manager is None:
        with _project_manager_lock:
            if _project_manager is None:
                _project_manager = ProjectManager()
    return _project_manager
