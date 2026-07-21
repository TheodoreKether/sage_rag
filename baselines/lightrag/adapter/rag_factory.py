"""Build a LightRAG instance from env / .env (no official source edits)."""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .paths import ENV_FILE, RAG_STORAGE_DIR

logger = logging.getLogger(__name__)

# Prefer local tiktoken cache (Azure blob downloads often fail in CN networks).
_TIKTOKEN_CACHE = Path(__file__).resolve().parents[1] / ".tiktoken_cache"
if _TIKTOKEN_CACHE.is_dir():
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE))


def load_baseline_env(env_file: Path | None = None) -> None:
    """Load baselines/lightrag/.env then process env (process wins)."""
    path = env_file or ENV_FILE
    if path.is_file():
        load_dotenv(dotenv_path=path, override=False)
        logger.info("Loaded env from %s", path)
    else:
        logger.debug("No .env at %s (using process environment)", path)


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _require_env(*names: str) -> str:
    value = _env(*names)
    if value:
        return value
    joined = " / ".join(names)
    raise RuntimeError(
        f"Missing required environment variable ({joined}). "
        f"Copy baselines/lightrag/.env.example to .env and fill values."
    )


async def create_rag(
    *,
    working_dir: Path | None = None,
    addon_params: dict[str, Any] | None = None,
):
    """Create and initialize LightRAG with OpenAI-compatible LLM + embedding."""
    # Import only after env is loaded so LightRAG sees correct defaults.
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    load_baseline_env()
    storage = Path(working_dir or RAG_STORAGE_DIR)
    storage.mkdir(parents=True, exist_ok=True)

    llm_model = _env("LLM_MODEL", default="gpt-4o-mini") or "gpt-4o-mini"
    llm_base = _env("LLM_BINDING_HOST", "OPENAI_BASE_URL")
    llm_key = _require_env("LLM_BINDING_API_KEY", "OPENAI_API_KEY")

    embed_model = _env("EMBEDDING_MODEL", default="text-embedding-3-small") or (
        "text-embedding-3-small"
    )
    embed_dim = int(_env("EMBEDDING_DIM", default="1536") or "1536")
    embed_base = _env(
        "EMBEDDING_BINDING_HOST",
        "LLM_BINDING_HOST",
        "OPENAI_BASE_URL",
    )
    embed_key = _env(
        "EMBEDDING_BINDING_API_KEY",
        "LLM_BINDING_API_KEY",
        "OPENAI_API_KEY",
    )
    if not embed_key:
        raise RuntimeError("Missing embedding API key")

    async def llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=llm_key,
            base_url=llm_base,
            **kwargs,
        )

    # Use .func to avoid double-wrapping EmbeddingFunc (LightRAG docs).
    embed_fn = partial(
        openai_embed.func,
        model=embed_model,
        api_key=embed_key,
        base_url=embed_base,
    )

    params = {
        "language": _env("SUMMARY_LANGUAGE", default="Chinese") or "Chinese",
    }
    if addon_params:
        params.update(addon_params)

    rag = LightRAG(
        working_dir=str(storage),
        llm_model_func=llm_model_func,
        llm_model_name=llm_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=int(
                _env("EMBEDDING_TOKEN_LIMIT", default="8192") or "8192"
            ),
            model_name=embed_model,
            func=embed_fn,
        ),
        addon_params=params,
        llm_model_max_async=int(_env("MAX_ASYNC_LLM", default="2") or "2"),
        embedding_func_max_async=int(
            _env("EMBEDDING_FUNC_MAX_ASYNC", default="4") or "4"
        ),
        # Must pass explicitly: LightRAG dataclass defaults read env at import time.
        default_llm_timeout=int(_env("LLM_TIMEOUT", default="240") or "240"),
        default_embedding_timeout=int(
            _env("EMBEDDING_TIMEOUT", default="180") or "180"
        ),
    )
    await rag.initialize_storages()
    logger.info(
        "LightRAG ready: working_dir=%s llm=%s embed=%s dim=%d",
        storage,
        llm_model,
        embed_model,
        embed_dim,
    )
    return rag
