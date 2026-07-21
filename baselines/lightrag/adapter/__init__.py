"""Sage RAG ↔ LightRAG adapter (does not modify official LightRAG sources)."""

from __future__ import annotations

__all__ = ["LightRAGRetriever"]


def __getattr__(name: str):
    if name == "LightRAGRetriever":
        from .retriever import LightRAGRetriever

        return LightRAGRetriever
    raise AttributeError(name)
