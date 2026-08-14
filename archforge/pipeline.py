"""
Orchestrates the three pillars into one request/response cycle, matching
the "How each answer is produced" flow: draft, then ground, then
validate. Nothing here leaves the process boundary except the one call
to the configured model endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ArchForgeConfig
from .llm_client import LLMClient
from .retrieval import RetrievalIndex
from .validator import SelfRepairValidator, ValidationResult

DRAFT_SYSTEM_PROMPT = (
    "You are a coding assistant fine-tuned on this team's own codebase and "
    "conventions. Write code that matches their existing style. If the "
    "provided reference material gives exact register names, values, or "
    "API signatures, use them precisely rather than inventing plausible-"
    "looking ones."
)


@dataclass
class PipelineResult:
    query: str
    draft: str
    grounded_answer: str
    final_code: str
    citations: list[str]
    validation: ValidationResult


class ArchForgePipeline:
    def __init__(self, config: ArchForgeConfig):
        self.config = config
        self.llm = LLMClient(config.model)
        self.index = RetrievalIndex(config.retrieval)
        self.validator = SelfRepairValidator(config.validation, llm_client=self.llm)

    def run(self, query: str) -> PipelineResult:
        # 1. DRAFT - first pass in the team's own style, no grounding yet.
        draft = self.llm.chat(DRAFT_SYSTEM_PROMPT, query)

        # 2. GROUND - pull relevant snippets from the team's own reference
        #    docs and give the model a chance to correct specifics against them.
        hits = self.index.search(query)
        citations = [chunk.source for chunk, _score in hits]

        if hits:
            reference_block = "\n\n".join(
                f"[{chunk.source}]\n{chunk.text}" for chunk, _score in hits
            )
            grounding_prompt = (
                f"Original request:\n{query}\n\n"
                f"Your first draft:\n{draft}\n\n"
                f"Reference material from the team's own documentation:\n"
                f"{reference_block}\n\n"
                "Revise the draft so every specific value, register name, or "
                "API detail matches the reference material exactly. If the "
                "draft was already correct, return it unchanged. Return only "
                "the code."
            )
            grounded_answer = self.llm.chat(DRAFT_SYSTEM_PROMPT, grounding_prompt)
        else:
            grounded_answer = draft

        # 3. VALIDATE - check against standards, self-repair, re-check.
        final_code, validation = self.validator.validate_and_repair(grounded_answer)

        return PipelineResult(
            query=query,
            draft=draft,
            grounded_answer=grounded_answer,
            final_code=final_code,
            citations=citations,
            validation=validation,
        )
