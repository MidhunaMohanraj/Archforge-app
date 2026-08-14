"""
Validate stage: check the drafted answer against coding standards, and
give the model a chance to repair its own output before the engineer
ever sees a violation.

The rule set here is a small, illustrative subset of MISRA C:2012 -
enough to demonstrate the pattern (static check -> feedback -> re-draft
-> re-check). A production deployment would plug in a real MISRA checker
(e.g. cppcheck with a MISRA addon, or a licensed tool) behind the same
`check()` interface; nothing else in the pipeline would need to change.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import ValidationConfig


@dataclass
class Violation:
    rule: str
    message: str
    line: int | None = None


@dataclass
class ValidationResult:
    passed: bool
    violations: list[Violation]
    attempts: int
    compile_checked: bool
    compile_output: str | None = None


class StandardsChecker:
    """
    Pattern-based subset of MISRA C:2012. Deliberately conservative: it
    flags clear, unambiguous violations and stays quiet otherwise, rather
    than drowning the engineer in false positives.
    """

    def check(self, code: str) -> list[Violation]:
        violations: list[Violation] = []
        lines = code.splitlines()

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            if re.search(r"\bgoto\b", stripped):
                violations.append(Violation(
                    rule="MISRA 15.1",
                    message="`goto` is not permitted.",
                    line=i,
                ))

            if re.search(r"\b(malloc|calloc|realloc|free)\s*\(", stripped):
                violations.append(Violation(
                    rule="MISRA 21.3",
                    message="Dynamic memory allocation is not permitted; "
                            "use static or pool allocation instead.",
                    line=i,
                ))

            single_line_block = re.match(
                r"^(if|else if|for|while)\s*\(.*\)\s*[^{;]+;?\s*$", stripped
            )
            if single_line_block and "{" not in stripped:
                violations.append(Violation(
                    rule="MISRA 15.6",
                    message="Compound statement must be enclosed in "
                            "braces, even for a single statement.",
                    line=i,
                ))

        violations.extend(self._check_switch_fallthrough(lines))
        return violations

    @staticmethod
    def _check_switch_fallthrough(lines: list[str]) -> list[Violation]:
        violations: list[Violation] = []
        in_case = False
        case_start_line = 0
        for i, raw in enumerate(lines, start=1):
            stripped = raw.strip()
            if re.match(r"^case\b|^default\s*:", stripped):
                in_case = True
                case_start_line = i
                continue
            if not in_case:
                continue
            if re.match(r"^(break|return|case\b|default\s*:|})", stripped):
                if re.match(r"^(case\b|default\s*:)", stripped):
                    continue
                in_case = False
            elif "fallthrough" in stripped.lower():
                in_case = False
        return violations


class SelfRepairValidator:
    """
    Runs the standards checker, and if the draft was already available
    from an LLM client, asks the model to fix each violation and
    re-checks. Stops as soon as the code is clean or the attempt budget
    runs out - whichever comes first.
    """

    def __init__(self, config: ValidationConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.checker = StandardsChecker()

    def validate_and_repair(self, code: str) -> tuple[str, ValidationResult]:
        attempts = 0
        current_code = code

        while True:
            violations = self.checker.check(current_code)
            attempts += 1

            if not violations:
                compile_ok, compile_output = self._maybe_compile(current_code)
                return current_code, ValidationResult(
                    passed=compile_ok,
                    violations=[],
                    attempts=attempts,
                    compile_checked=self.config.run_compile_check,
                    compile_output=compile_output,
                )

            if attempts > self.config.max_repair_attempts or self.llm_client is None:
                compile_ok, compile_output = self._maybe_compile(current_code)
                return current_code, ValidationResult(
                    passed=False,
                    violations=violations,
                    attempts=attempts,
                    compile_checked=self.config.run_compile_check,
                    compile_output=compile_output,
                )

            current_code = self._repair(current_code, violations)

    def _repair(self, code: str, violations: list[Violation]) -> str:
        issue_list = "\n".join(
            f"- Line {v.line}, {v.rule}: {v.message}" for v in violations
        )
        prompt = (
            "The following code fails these standards checks:\n\n"
            f"{issue_list}\n\n"
            "Fix every issue listed and return only the corrected code, "
            "with no explanation.\n\n"
            f"```\n{code}\n```"
        )
        return self.llm_client.chat(
            system_prompt="You are fixing MISRA C violations in existing code. "
                          "Preserve behaviour and style; change only what is needed "
                          "to resolve the listed issues.",
            user_prompt=prompt,
        )

    def _maybe_compile(self, code: str) -> tuple[bool, str | None]:
        if not self.config.run_compile_check:
            return True, None
        if shutil.which(self.config.compiler) is None:
            return True, f"{self.config.compiler} not found on PATH; compile check skipped."

        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                [self.config.compiler, "-fsyntax-only", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0, result.stderr
        except (subprocess.SubprocessError, OSError) as exc:
            return False, str(exc)
        finally:
            temp_path.unlink(missing_ok=True)
