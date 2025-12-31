from contextlib import contextmanager
from time import perf_counter
from typing import Optional

from rag.config import logger

@contextmanager
def log_timing(label: str, **extra):
  start = perf_counter()
  try:
    yield
  finally:
    elapsed = perf_counter() - start
    payload = {"elapsed_ms": round(elapsed * 1000, 2)}
    if extra:
      payload.update(extra)
    logger.info(label, extra=payload)
