"""LLM client abstraction using litellm."""

from __future__ import annotations

import asyncio
import json
import logging
import re

import litellm

from vuln_triage.config import LLMConfig
from vuln_triage.sarif.models import Finding, TriageVerdict

from .prompts import (
    EVAL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_eval_prompt,
    build_triage_prompt,
)

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMError(Exception):
    """Raised when LLM call fails."""


class LLMClient:
    """LLM client for vulnerability triage via litellm."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    async def triage_finding(
        self,
        finding: Finding,
        code_snippet: str,
        codeql_context: str | None = None,
    ) -> tuple[TriageVerdict, int]:
        """Send a single finding to the LLM for triage.

        Returns (TriageVerdict, tokens_used).
        """
        prompt = build_triage_prompt(finding, code_snippet, codeql_context)

        async with self._semaphore:
            try:
                response = await litellm.acompletion(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                raise LLMError(f"LLM call failed: {e}") from e

        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        verdict = self._parse_response(content)
        return verdict, tokens

    async def classify_code(
        self,
        code_snippet: str,
        codeql_context: str | None = None,
    ) -> tuple[TriageVerdict, int]:
        """Classify a standalone code snippet (eval / benchmark mode).

        Uses eval-specific prompts without SARIF metadata.
        Returns (TriageVerdict, tokens_used).
        """
        prompt = build_eval_prompt(code_snippet, codeql_context)

        async with self._semaphore:
            try:
                response = await litellm.acompletion(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                raise LLMError(f"LLM call failed: {e}") from e

        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        verdict = self._parse_response(content)
        return verdict, tokens

    def triage_finding_sync(
        self,
        finding: Finding,
        code_snippet: str,
        codeql_context: str | None = None,
    ) -> tuple[TriageVerdict, int]:
        """Synchronous wrapper around triage_finding."""
        return asyncio.run(self.triage_finding(finding, code_snippet, codeql_context))

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Try to extract a JSON object from arbitrary text.

        Handles cases where the model wraps JSON in markdown fences or
        writes prose before/after the JSON object.
        """
        # 1. Strip markdown fences first
        stripped = text.strip()
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                stripped = stripped[first_nl + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()
            # After stripping fences, it might be clean JSON
            try:
                json.loads(stripped)
                return stripped
            except json.JSONDecodeError:
                pass

        # 2. Try the raw text as-is
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

        # 3. Find a JSON object containing "verdict" anywhere in the text
        # Match outermost { ... } that contains "verdict"
        for match in re.finditer(r"\{[^{}]*\"verdict\"[^{}]*\}", text):
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

        # 4. Greedy: find the largest { ... } block
        brace_start = text.find("{")
        if brace_start != -1:
            brace_end = text.rfind("}")
            if brace_end > brace_start:
                candidate = text[brace_start : brace_end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        return None

    def _parse_response(self, content: str) -> TriageVerdict:
        """Parse the LLM's JSON response into a TriageVerdict."""
        extracted = self._extract_json(content)
        if not extracted:
            logger.error("Failed to extract JSON from LLM response: %s", content[:200])
            return TriageVerdict(
                verdict="uncertain",
                confidence=0.0,
                reasoning=f"Failed to parse LLM response. Raw: {content[:500]}",
            )

        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse extracted JSON: %s", extracted[:200])
            return TriageVerdict(
                verdict="uncertain",
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {e}. Raw: {content[:500]}",
            )

        verdict_str = data.get("verdict", "uncertain")
        if verdict_str not in ("true_positive", "false_positive", "uncertain"):
            logger.warning("Unknown verdict '%s', defaulting to uncertain", verdict_str)
            verdict_str = "uncertain"

        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        reasoning = data.get("reasoning", "No reasoning provided")

        return TriageVerdict(
            verdict=verdict_str,
            confidence=confidence,
            reasoning=reasoning,
        )
