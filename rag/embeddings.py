"""
Embedding model factory.

Supports OpenAI and Gemini embedding models.
"""

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from rag.config import settings, require_openai_key, get_gemini_key, logger


def get_embedding_model() -> BaseEmbedding:
    """
    Get embedding model based on configuration.
    
    Returns:
        BaseEmbedding instance
        
    Raises:
        ValueError: If provider is unsupported or API key is missing
    """
    provider = settings.embedding_provider.lower()
    model = settings.embedding_model
    
    if provider == "openai":
        require_openai_key()
        logger.info(f"Using OpenAI embeddings: {model}")
        return OpenAIEmbedding(model=model)
    
    elif provider == "gemini":
        from google import genai
        from google.genai.types import EmbedContentConfig
        
        api_key = get_gemini_key()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it: export GEMINI_API_KEY='your-key'"
            )
        
        logger.info(f"Using Gemini embeddings: {model}")
        
        class GeminiEmbedding(BaseEmbedding):
            """Gemini embedding wrapper for LlamaIndex."""
            
            _model: str
            _client: object
            
            def __init__(self, model: str, api_key: str):
                super().__init__(model_name=model)
                self._model = model
                self._client = genai.Client(api_key=api_key)
            
            def _get_query_embedding(self, query: str) -> list[float]:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=query
                )
                return response.embeddings[0].values
            
            def _get_text_embedding(self, text: str) -> list[float]:
                return self._get_query_embedding(text)
            
            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                """Batch embedding for better performance."""
                if not texts:
                    return []
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=texts
                )
                return [emb.values for emb in response.embeddings]
            
            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)
            
            async def _aget_text_embedding(self, text: str) -> list[float]:
                return self._get_text_embedding(text)
        
        return GeminiEmbedding(model=model, api_key=api_key)
    
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported: openai, gemini"
        )
