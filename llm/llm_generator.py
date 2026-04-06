"""
LLM generator optimized for fast responses using Ollama.
"""

import os
from typing import Any, Generator

import requests

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:latest")
CHAT_URL = f"{OLLAMA_BASE}/api/chat"


def _clinical_system_prompt() -> str:
    """Concise system prompt for faster generation."""
    return (
        "You are a clinical data assistant interpreting federated learning results. "
        "Be concise and direct. Use 2-4 sentences per response. "
        "Base answers on provided metrics and data context only. "
        "No disclaimers or caveats unless asked."
    )


def _run_metrics_only(
    accuracy: float,
    loss: float,
    num_rounds: int,
    accuracy_history: list[float] | None,
) -> str:
    """Compact metrics summary."""
    lines = [
        f"FL Results: {num_rounds} rounds, {accuracy*100:.1f}% accuracy, {loss:.4f} loss."
    ]
    if accuracy_history and len(accuracy_history) > 1:
        trend = accuracy_history[-1] - accuracy_history[0]
        lines.append(f"Trend: {trend*100:+.1f}pp from round 1 to {len(accuracy_history)}.")
    return " ".join(lines)


def build_initial_clinical_messages(
    accuracy: float,
    loss: float,
    num_rounds: int = 0,
    accuracy_history: list[float] | None = None,
    *,
    data_context: str | None = None,
) -> list[dict[str, str]]:
    """Build initial messages for chat."""
    parts = [_run_metrics_only(accuracy, loss, num_rounds, accuracy_history)]
    if data_context:
        parts.append(data_context.strip())
    parts.append("Give a brief interpretation (2-4 sentences): is performance promising, convergence trend, one suggestion.")
    return [
        {"role": "system", "content": _clinical_system_prompt()},
        {"role": "user", "content": " ".join(parts)},
    ]


def ollama_chat(
    messages: list[dict[str, str]],
    *,
    timeout: int = 60,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> tuple[str | None, str | None]:
    """
    Ollama chat with optimized parameters.
    Returns (assistant_content, error_message).
    """
    try:
        response = requests.post(
            CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                }
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        msg = data.get("message") or {}
        content = msg.get("content", "").strip()
        
        if not content:
            return None, "No response from LLM."
        return content, None
    except requests.ConnectionError:
        return None, f"Cannot connect to Ollama. Run: ollama serve && ollama pull {OLLAMA_MODEL}"
    except requests.Timeout:
        return None, "LLM timeout. Try a smaller model."
    except requests.HTTPError as e:
        return None, f"LLM error: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def ollama_chat_stream(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Streaming chat for real-time output display.
    Yields text chunks as they arrive.
    """
    try:
        response = requests.post(
            CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                }
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        
        import json
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield f"\n[Error: {e}]"


def generate_insight(
    accuracy: float,
    loss: float,
    num_rounds: int = 0,
    accuracy_history: list[float] | None = None,
    *,
    data_context: str | None = None,
) -> str:
    """Single-shot insight generation."""
    messages = build_initial_clinical_messages(
        accuracy, loss, num_rounds, accuracy_history, data_context=data_context
    )
    content, err = ollama_chat(messages, max_tokens=250)
    if err:
        return f"**{err}**"
    return content or "**No response from LLM.**"


def generate_insight_with_history(
    accuracy: float,
    loss: float,
    num_rounds: int = 0,
    accuracy_history: list[float] | None = None,
    *,
    data_context: str | None = None,
) -> tuple[list[dict[str, str]] | None, str | None]:
    """
    Generate first insight and return full message history.
    Returns (messages_with_assistant_reply, error).
    """
    messages = build_initial_clinical_messages(
        accuracy, loss, num_rounds, accuracy_history, data_context=data_context
    )
    content, err = ollama_chat(messages, max_tokens=250)
    if err:
        return None, err
    if not content:
        return None, "Empty response from LLM."
    out = list(messages)
    out.append({"role": "assistant", "content": content})
    return out, None


def quick_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 150,
) -> tuple[str | None, str | None]:
    """Fast follow-up chat with shorter responses."""
    return ollama_chat(messages, max_tokens=max_tokens, temperature=0.3)


def stream_initial_insight(
    accuracy: float,
    loss: float,
    num_rounds: int = 0,
    accuracy_history: list[float] | None = None,
    *,
    data_context: str | None = None,
) -> Generator[tuple[str, list[dict[str, str]] | None], None, None]:
    """
    Stream the initial insight generation.
    Yields (chunk, None) during streaming, then (final_chunk, full_messages) at end.
    """
    messages = build_initial_clinical_messages(
        accuracy, loss, num_rounds, accuracy_history, data_context=data_context
    )
    
    full_response = ""
    for chunk in ollama_chat_stream(messages, max_tokens=250):
        full_response += chunk
        yield chunk, None
    
    if full_response and "[Error:" not in full_response:
        messages.append({"role": "assistant", "content": full_response})
        yield "", messages
    else:
        yield "", None
