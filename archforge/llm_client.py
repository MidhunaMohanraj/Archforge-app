"""
Thin wrapper around an OpenAI-compatible chat endpoint.

Kept deliberately small: this is the only place in the codebase that talks
to the network, and it only ever talks to the single endpoint the site
configures (their own on-premise server in production). If that endpoint
is unreachable - for example during a demo with no model server running -
the client falls back to a clearly-labelled offline stub so the rest of
the pipeline can still be exercised.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import ModelConfig


class LLMClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            url=f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read())
                return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
            return self._offline_stub(user_prompt)

    def _offline_stub(self, user_prompt: str) -> str:
        """
        Used only when no model endpoint is reachable, so the pipeline can
        still be demoed end to end. Clearly marked - this is not a real
        answer and should never be mistaken for one.
        """
        return (
            "// [offline stub - no model endpoint reachable]\n"
            "// Configure archforge.config.json with a running "
            "OpenAI-compatible endpoint to get a real draft.\n"
            f"// Prompt received: {user_prompt.strip()[:200]}"
        )
