from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_chat_model(provider: str | None = None, **kwargs: Any):
    """Return a LangChain chat model instance for the configured provider."""
    provider = (provider or os.getenv("AI_PROVIDER", "openai")).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
            temperature=float(kwargs.get("temperature", os.getenv("MODEL_TEMPERATURE", "0.2"))),
            max_tokens=kwargs.get("max_tokens", None),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=kwargs.get("model", os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")),
            api_key=kwargs.get("api_key", os.getenv("ANTHROPIC_API_KEY")),
            temperature=float(kwargs.get("temperature", os.getenv("MODEL_TEMPERATURE", "0.2"))),
            max_tokens=kwargs.get("max_tokens", None),
        )

    if provider in {"google", "gemini", "google-gemini"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=kwargs.get("model", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")),
            api_key=kwargs.get("api_key", os.getenv("GOOGLE_API_KEY")),
            temperature=float(kwargs.get("temperature", os.getenv("MODEL_TEMPERATURE", "0.2"))),
            max_output_tokens=kwargs.get("max_tokens", None),
        )

    raise ValueError(f"Unsupported AI provider: {provider}. Use 'openai', 'anthropic', or 'google'.")


def get_embeddings(provider: str | None = None, **kwargs: Any):
    """Return an embedding model instance for the configured provider."""
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "openai")).lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=kwargs.get("model", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=kwargs.get("model_name", os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")),
        )

    if provider in {"google", "gemini", "google-gemini"}:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=kwargs.get("model", os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")),
            api_key=kwargs.get("api_key", os.getenv("GOOGLE_API_KEY")),
        )

    raise ValueError(f"Unsupported embedding provider: {provider}. Use 'openai', 'huggingface', or 'google'.")


def main() -> None:
    """Simple project starter entry point."""
    provider = os.getenv("AI_PROVIDER", "openai")
    model = get_chat_model(provider)
    print(f"AI app initialized with provider: {provider}")
    print(f"Chat model class: {type(model).__name__}")


__all__ = ["main", "get_chat_model", "get_embeddings"]
