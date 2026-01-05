"""
Centralized configuration for RAG system.

All constants, settings, and configuration are defined here.
"""

import os
import threading
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, Dict, List, FrozenSet


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

_setup_lock = threading.Lock()
_logger_initialized = False


def setup_logging(level: str = None, force: bool = False) -> logging.Logger:
    """
    Configure and return the RAG logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        force: If True, force reconfiguration even if already configured

    Returns:
        logging.Logger instance
    """
    global _logger_initialized
    log_level = level or os.getenv("RAG_LOG_LEVEL", "WARNING")

    with _setup_lock:
        if _logger_initialized and not force:
            return logging.getLogger("rag")

        # Remove existing rag handlers to allow reconfiguration
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if "rag" in str(getattr(handler, 'name', '')):
                root_logger.removeHandler(handler)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=force
        )
        _logger_initialized = True
        return logging.getLogger("rag")


logger = setup_logging()


# ---------------------------------------------------------------------------
# File Extensions
# ---------------------------------------------------------------------------

SUPPORTED_CODE_EXTS: FrozenSet[str] = frozenset({
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.sh', '.bash', '.zsh', '.sql', '.yaml', '.yml', '.toml', '.dockerfile',
    '.makefile', '.vue', '.svelte', '.astro', '.html', '.css'
})

SUPPORTED_DOC_EXTS: FrozenSet[str] = frozenset({
    '.txt', '.pdf', '.docx', '.doc', '.md', '.epub',
    '.ppt', '.pptx', '.pptm', '.xls', '.xlsx', '.csv', '.json', '.xml',
    '.ipynb', '.hwp', '.mbox', '.rtf'
})

IMAGE_EXTS: FrozenSet[str] = frozenset({'.jpg', '.jpeg', '.png'})

ALLOWED_EXTENSIONS: FrozenSet[str] = SUPPORTED_CODE_EXTS | SUPPORTED_DOC_EXTS | IMAGE_EXTS


# ---------------------------------------------------------------------------
# Language Mapping for Code Splitting (Complete)
# ---------------------------------------------------------------------------

CODE_LANGUAGE_MAP: Dict[str, str] = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cs': 'c_sharp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.html': 'html',
    '.css': 'css',
    '.php': 'php',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.r': 'r',
    '.sh': 'shell',
    '.bash': 'shell',
    '.zsh': 'shell',
    '.sql': 'sql',
    '.toml': 'toml',
    '.dockerfile': 'dockerfile',
    '.makefile': 'makefile',
    '.vue': 'vue',
    '.svelte': 'svelte',
    '.astro': 'astro',
}

LANGUAGE_TO_EXTENSIONS: Dict[str, List[str]] = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "cpp": [".cpp", ".hpp", ".cc", ".cxx", ".c", ".h"],
    "csharp": [".cs"],
    "go": [".go"],
    "rust": [".rs"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt"],
    "scala": [".scala"],
    "shell": [".sh", ".bash", ".zsh"],
    "sql": [".sql"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "yaml": [".yaml", ".yml"],
    "json": [".json"],
    "xml": [".xml"],
    "markdown": [".md", ".markdown"],
    "vue": [".vue"],
    "svelte": [".svelte"],
    "astro": [".astro"],
    "dockerfile": [".dockerfile"],
    "makefile": [".makefile", "Makefile"],
}


# ---------------------------------------------------------------------------
# Default Exclusion Patterns
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDES: FrozenSet[str] = frozenset({
    # Package managers
    "node_modules", "__pycache__", ".git", ".svn", ".hg",
    # Virtual environments
    "venv", "env", ".venv", ".env",
    # Build outputs
    "build", "dist", "target", "out",
    # IDE files
    ".idea", ".vscode", ".vs",
    # Compiled files
    "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll",
    # System files
    ".DS_Store", "Thumbs.db"
})


# ---------------------------------------------------------------------------
# Project Detection Markers
# ---------------------------------------------------------------------------

CODE_INDICATOR_FILES: FrozenSet[str] = frozenset({
    'package.json', 'requirements.txt', 'requirements-dev.txt',
    'pyproject.toml', 'setup.py', 'setup.cfg', 'Pipfile', 'Pipfile.lock',
    'poetry.lock', 'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle',
    'Makefile', 'CMakeLists.txt', 'environment.yml'
})

CODE_INDICATOR_DIRS: FrozenSet[str] = frozenset({
    '.git', '.vscode', '.idea', '.github', 'src', 'lib', 'include',
    'apps', 'services', 'packages', 'node_modules', 'venv'
})


# ---------------------------------------------------------------------------
# Settings Class
# ---------------------------------------------------------------------------

VALID_EMBEDDING_PROVIDERS: FrozenSet[str] = frozenset({"openai", "gemini"})


@dataclass
class RAGSettings:
    """Central configuration for RAG system."""

    # Paths
    script_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.absolute())
    storage_path: Path = field(init=False)

    # Embedding
    embedding_provider: str = field(init=False)
    embedding_model: str = field(init=False)

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 200
    code_chunk_lines: int = 40
    code_chunk_overlap: int = 15
    code_max_chars: int = 1500

    # Retrieval
    min_top_k: int = 1
    max_top_k: int = 50
    default_top_k: int = 6
    rerank_candidate_multiplier: int = 2
    min_rerank_candidates: int = 10

    # Thresholds
    low_score_threshold: float = 0.2
    rerank_delta_threshold: float = 0.05
    rerank_min_results: int = 3
    hyde_trigger_min_results: int = 1
    hyde_trigger_score: float = 0.1
    # HyDE model configuration: can be customized via environment variable or settings override
    # Example models: "gpt-3.5-turbo" (default, fast), "gpt-4" (better quality, slower)
    hyde_model: str = "gpt-3.5-turbo"
    hyde_timeout: float = 30.0
    hyde_max_retries: int = 2
    hyde_initial_backoff: float = 0.5

    # API Server
    request_buffer_size: int = 200
    log_buffer_size: int = 400

    # Locking
    lock_retry_attempts: int = 3
    lock_retry_delay: float = 0.1

    # File constraints
    max_file_size_mb: int = 100
    max_query_length: int = 10000

    # Project defaults
    default_project: str = "rag_collection"
    project_metadata_filename: str = "project_metadata.json"
    ingest_manifest_filename: str = "ingest_manifest.json"
    tracking_filename: str = "indexed_files.json"

    def __post_init__(self):
        self.storage_path = self.script_dir / "storage"

        # Validate embedding provider
        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        if provider not in VALID_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: {provider}. "
                f"Valid providers: {VALID_EMBEDDING_PROVIDERS}"
            )
        self.embedding_provider = provider
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


# Global settings instance
settings = RAGSettings()


# ---------------------------------------------------------------------------
# Environment Validation
# ---------------------------------------------------------------------------

def validate_api_key(key: str | None, key_name: str, min_length: int = 20) -> str:
    """
    Validate API key format and existence.

    Args:
        key: API key string or None
        key_name: Name of the key for error messages
        min_length: Minimum expected length for basic validation

    Returns:
        Validated key string

    Raises:
        ValueError: If key is missing or invalid
    """
    if not key:
        raise ValueError(
            f"{key_name} not found in environment variables.\n"
            f"Please set it: export {key_name}='your-api-key'"
        )

    # Basic format validation
    key = key.strip()
    if len(key) < min_length:
        raise ValueError(
            f"{key_name} appears invalid (too short, < {min_length} characters).\n"
            f"Please check your API key."
        )

    return key


def require_openai_key() -> str:
    """Get and validate OpenAI API key or raise ValueError."""
    key = os.getenv("OPENAI_API_KEY")
    return validate_api_key(key, "OPENAI_API_KEY", min_length=20)


def get_cohere_key() -> str | None:
    """Get and validate Cohere API key (optional)."""
    key = os.getenv("COHERE_API_KEY")
    if key:
        # Validate if provided (warn but don't fail)
        try:
            return validate_api_key(key, "COHERE_API_KEY", min_length=20)
        except ValueError as e:
            logger.warning(f"Cohere API key validation failed: {e}")
            return key
    return None


def get_github_token() -> str | None:
    """Get GitHub token (optional)."""
    return os.getenv("GITHUB_TOKEN")


def get_firecrawl_key() -> str | None:
    """Get and validate Firecrawl API key (optional)."""
    key = os.getenv("FIRECRAWL_API_KEY")
    if key:
        # Validate if provided
        try:
            return validate_api_key(key, "FIRECRAWL_API_KEY", min_length=20)
        except ValueError as e:
            logger.warning(f"Firecrawl API key validation failed: {e}")
            return key
    return None


def get_gemini_key() -> str | None:
    """Get and validate Gemini API key (optional)."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        # Validate if provided
        try:
            return validate_api_key(key, "GEMINI_API_KEY", min_length=20)
        except ValueError as e:
            logger.warning(f"Gemini API key validation failed: {e}")
            return key
    return None


def validate_required_keys() -> dict[str, str]:
    """
    Validate all required API keys at startup.

    Returns:
        Dict with validation status

    Raises:
        ValueError: If required keys are missing/invalid
    """
    errors = {}

    # Always require OpenAI key (used for embeddings + HyDE)
    try:
        require_openai_key()
    except ValueError as e:
        errors["OPENAI_API_KEY"] = str(e)

    # Warn if embedding provider is Gemini but key is missing
    if settings.embedding_provider == "gemini":
        try:
            gemini_key = get_gemini_key()
            if not gemini_key:
                errors["GEMINI_API_KEY"] = (
                    "GEMINI_API_KEY required when EMBEDDING_PROVIDER=gemini.\n"
                    "Please set it: export GEMINI_API_KEY='your-api-key'"
                )
        except Exception as e:
            errors["GEMINI_API_KEY"] = str(e)

    if errors:
        error_msg = "API key validation failed:\n\n" + "\n".join(
            f"- {k}: {v}" for k, v in errors.items()
        )
        raise ValueError(error_msg)

    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "openai_key": "validated"
    }
