from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

NO_INFO_RESPONSE = "I could not find the answer in the provided documents."
logger = logging.getLogger(__name__)


def _clip_text(value: str, max_len: int = 500) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len]}..."


def _trim_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()

    return str(content).strip()


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [p.strip() for p in parts if p and p.strip()]


def _is_no_info_answer(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    markers = (
        "i could not find the answer in the provided documents",
        "answer is not present in the provided documents",
        "answer is not in the provided documents",
        "answer is not present in the context",
        "cannot find the answer in the provided documents",
    )
    return any(marker in normalized for marker in markers)


def _extractive_answer(query: str, context_chunks: list[str]) -> str:
    query_terms = {w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(w) > 2}
    sentences: list[str] = []
    for chunk in context_chunks:
        sentences.extend(_split_sentences(chunk))

    if not sentences:
        return NO_INFO_RESPONSE

    best_sentence = sentences[0]
    best_score = -1
    for sentence in sentences:
        sent_terms = {w for w in re.findall(r"[a-zA-Z0-9]+", sentence.lower()) if len(w) > 2}
        score = len(query_terms & sent_terms)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI | None:
    settings = get_settings()
    api_key = settings.openai_api_key or settings.llm_api_key
    if not api_key:
        return None
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        base_url=settings.openai_base_url or None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def answer_query(query: str, context_chunks: list[str]) -> str:
    settings = get_settings()
    logger.info(
        "LLM request started: provider=%s model=%s query=%r context_chunks=%s",
        settings.llm_provider,
        settings.openai_model,
        query,
        len(context_chunks),
    )
    if settings.llm_provider != "openai":
        context_preview = " | ".join(context_chunks[:3])
        logger.info("LLM provider is not openai; returning dummy response. context_preview=%r", _clip_text(context_preview))
        return f"[dummy-llm] Answer for '{query}'. Context: {context_preview}" if context_preview else f"[dummy-llm] Answer for '{query}'."

    model = get_chat_model()
    if model is None:
        logger.warning("LLM request aborted: OpenAI provider selected but API key missing")
        return "OpenAI provider selected, but OPENAI_API_KEY is missing."

    context = "\n\n".join(context_chunks) if context_chunks else "No retrieved context."
    try:
        system_prompt = settings.llm_system_prompt.format(context=context, question=query)
    except Exception:
        system_prompt = settings.llm_system_prompt
    logger.info(
        "LLM request payload: query=%r context_preview=%r system_prompt_preview=%r",
        query,
        _clip_text(context, 700),
        _clip_text(system_prompt, 700),
    )

    started = time.monotonic()
    try:
        response = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]
        )
        logger.info(f"model response: {response}")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        answer = _extract_text(response.content)
        logger.info(
            "LLM response received: elapsed_ms=%s raw_response_preview=%r",
            elapsed_ms,
            _clip_text(answer, 700),
        )
        if not answer:
            logger.warning("LLM response empty; returning NO_INFO_RESPONSE")
            return NO_INFO_RESPONSE
        answer = _trim_words(answer, settings.llm_max_answer_words)
        if _is_no_info_answer(answer) and context_chunks:
            extractive = _trim_words(_extractive_answer(query, context_chunks), settings.llm_max_answer_words)
            logger.info(
                "LLM returned no-info marker; using extractive fallback. extractive_preview=%r",
                _clip_text(extractive, 300),
            )
            return extractive
        logger.info("LLM final answer: %r", _clip_text(answer, 400))
        return answer
    except Exception:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.exception("LLM call failed: elapsed_ms=%s", elapsed_ms)
        context_preview = " | ".join(context_chunks[:3])
        return (
            f"[fallback-llm] Answer for '{query}'. Context: {context_preview}"
            if context_preview
            else f"[fallback-llm] Answer for '{query}'."
        )
