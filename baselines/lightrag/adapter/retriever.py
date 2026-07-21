"""RetrieverBase adapter around LightRAG.aquery_data."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from .id_map import load_chunk_to_unit
from .paths import CHUNK_TO_UNIT_MAP, RAG_STORAGE_DIR, REPO_ROOT
from .rag_factory import create_rag, load_baseline_env

logger = logging.getLogger(__name__)


def _load_retriever_base():
    """Load retriever_base.py without importing src.retrieval package (avoids bm25/faiss)."""
    module_name = "sage_rag_retriever_base"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = REPO_ROOT / "src" / "retrieval" / "retriever_base.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_rb = _load_retriever_base()
EvidenceUnit = _rb.EvidenceUnit
RetrieverBase = _rb.RetrieverBase


class LightRAGRetriever(RetrieverBase):
    """Map LightRAG structured retrieval results back to Evidence Units."""

    def __init__(
        self,
        *,
        working_dir: Path | str | None = None,
        chunk_map_path: Path | str | None = None,
        mode: str = "mix",
        enable_rerank: bool = False,
        rag=None,
    ) -> None:
        load_baseline_env()
        self.working_dir = Path(working_dir or RAG_STORAGE_DIR)
        self.chunk_map_path = Path(chunk_map_path or CHUNK_TO_UNIT_MAP)
        self.mode = mode
        self.enable_rerank = enable_rerank
        self._chunk_to_unit = load_chunk_to_unit(self.chunk_map_path)
        self._rag = rag
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owns_loop = False

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._owns_loop = True
        self._loop = loop
        return loop

    def _run(self, coro):
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None and running.is_running():
            # Called from inside an already-running loop (e.g. async smoke test).
            import nest_asyncio

            nest_asyncio.apply()
            return running.run_until_complete(coro)

        loop = self._get_loop()
        return loop.run_until_complete(coro)

    async def _ensure_rag(self):
        if self._rag is None:
            self._rag = await create_rag(working_dir=self.working_dir)
        return self._rag

    def close(self) -> None:
        if self._rag is not None:
            try:
                self._run(self._rag.finalize_storages())
            except Exception as exc:
                logger.warning("finalize_storages failed: %s", exc)
            self._rag = None

    def retrieve(self, query: str, top_k: int) -> list[EvidenceUnit]:
        return self._run(self._aretrieve(query, top_k))

    async def _aretrieve(self, query: str, top_k: int) -> list[EvidenceUnit]:
        from lightrag import QueryParam

        rag = await self._ensure_rag()
        param = QueryParam(
            mode=self.mode,  # type: ignore[arg-type]
            chunk_top_k=top_k,
            top_k=top_k,
            enable_rerank=self.enable_rerank,
        )
        result = await rag.aquery_data(query, param=param)
        return self._parse_result(result, top_k=top_k)

    def _parse_result(self, result: dict[str, Any], *, top_k: int) -> list[EvidenceUnit]:
        if not result or result.get("status") == "failure":
            logger.debug(
                "LightRAG query failure/empty: %s",
                result.get("message") if isinstance(result, dict) else result,
            )
            return []

        data = result.get("data") or {}
        chunks = data.get("chunks") or []
        units: list[EvidenceUnit] = []
        seen: set[str] = set()

        for _rank, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            unit_id = self._chunk_to_unit.get(chunk_id)
            if not unit_id:
                file_path = str(chunk.get("file_path") or "").strip()
                unit_id = self._chunk_to_unit.get(file_path) or (
                    file_path if file_path in self._chunk_to_unit.values() else ""
                )
            if not unit_id or unit_id in seen:
                continue
            seen.add(unit_id)
            document_id = unit_id.split("::", 1)[0] if "::" in unit_id else ""
            units.append(
                EvidenceUnit(
                    unit_id=unit_id,
                    document_id=document_id,
                    parent_clause="",
                    text=str(chunk.get("content") or ""),
                    metadata={
                        "lightrag_chunk_id": chunk_id,
                        "lightrag_reference_id": chunk.get("reference_id"),
                        "lightrag_file_path": chunk.get("file_path"),
                        "lightrag_mode": self.mode,
                    },
                    rank=len(units) + 1,
                    score=None,
                )
            )
            if len(units) >= top_k:
                break

        return units
