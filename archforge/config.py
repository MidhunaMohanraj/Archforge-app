"""
Configuration for an ArchForge deployment.

Everything a site would actually change lives here: which model endpoint
to draft from, where the reference documents are, and which standards
checks to enforce. Nothing about the pipeline logic itself should need to
change between deployments - only these values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("archforge.config.json")

@dataclass
class ModelConfig:
    # Any OpenAI-compatible endpoint works here: a local vLLM server, Ollama,
    # or a hosted API used for early testing before the on-premise box is up.
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed-for-local-serving"
    model_name: str = "archforge-domain-model"
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class RetrievalConfig:
    docs_dir: str = "sample_docs"
    top_k: int = 4
    # Minimum similarity score before a snippet is considered relevant
    # enough to ground an answer. Below this, the pipeline says so rather
    # than pretending it found something.
    min_score: float = 0.05


@dataclass
class ValidationConfig:
    ruleset: str = "misra-c-2012-subset"
    max_repair_attempts: int = 2
    run_compile_check: bool = False
    compiler: str = "gcc"


@dataclass
class ArchForgeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "ArchForgeConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            validation=ValidationConfig(**raw.get("validation", {})),
        )

    def save(self, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2))
