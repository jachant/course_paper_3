"""Prompt templates for LLM-based vulnerability triage."""

from __future__ import annotations

from vuln_triage.sarif.models import Finding

SYSTEM_PROMPT = """\
You are a senior application security engineer performing vulnerability triage.

Your task is to analyze findings from a SAST (Static Application Security Testing) tool \
and determine whether each finding is a TRUE POSITIVE (a real, exploitable vulnerability) \
or a FALSE POSITIVE (not actually exploitable or not a real security issue).

You will be given:
1. SAST finding metadata (rule, severity, description)
2. The relevant source code with the flagged line(s) marked with >>>
3. CodeQL graph analysis when available, which may include:
   - TAINT FLOW PATHS: full data flow chains from source to sink
   - SANITIZERS & GUARDS: barrier functions (escape, validate, isinstance, etc.)
   - ENCLOSING SCOPE: the function/method containing the finding with decorators
   - CALL GRAPH: callers and callees of the enclosing function
   - SINK CLASSIFICATION: type of dangerous operation (SqlExecution, CommandExecution, etc.)
   - SOURCE CLASSIFICATION: type of data source (RemoteFlowSource, EnvironmentSource, etc.)

Guidelines:
- Focus on whether tainted data from user input actually reaches the vulnerable sink
- If TAINT FLOW PATHS are present, they are the strongest signal — trace the full chain
- If SANITIZERS are present, check whether they effectively neutralize the threat
- SINK CLASSIFICATION tells you the attack type — verify the sink is actually dangerous
- SOURCE CLASSIFICATION tells you attacker controllability — RemoteFlowSource is high risk
- Check for framework-level protections (ORM parameterized queries, template auto-escaping, etc.)
- Consider whether the code is reachable in practice (not dead code, test-only code, etc.)
- If the finding is in test code, it is almost always a false positive
- When CodeQL shows no taint path from user input to the sink, lean toward false positive
- When uncertain, explain what additional information would help determine the verdict

Always respond with valid JSON. No markdown, no code fences — just the JSON object.\
"""

USER_PROMPT_TEMPLATE = """\
## SAST Finding

- **Rule**: {rule_id}
- **Tool**: {tool_name}
- **Severity**: {severity}
- **Level**: {level}
- **Message**: {message}
- **Location**: {file_path}:{start_line}

## Source Code

```python
{code_snippet}
```

{graph_section}

## Task

Analyze this finding and classify it. Respond with this JSON structure:

{{
  "verdict": "true_positive" | "false_positive" | "uncertain",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<detailed explanation of your analysis>"
}}\
"""


def build_triage_prompt(
    finding: Finding,
    code_snippet: str,
    codeql_context: str | None = None,
) -> str:
    """Build the user prompt for a single finding triage."""
    if codeql_context:
        graph_section = f"## CodeQL Graph Analysis\n\n```\n{codeql_context}\n```"
    else:
        graph_section = (
            "## CodeQL Graph Analysis\n\n"
            "_CodeQL analysis unavailable for this finding. "
            "Base your analysis on the source code alone._"
        )

    return USER_PROMPT_TEMPLATE.format(
        rule_id=finding.rule_id,
        tool_name=finding.tool_name or "unknown",
        severity=finding.severity,
        level=finding.level,
        message=finding.message,
        file_path=finding.file_path,
        start_line=finding.start_line,
        code_snippet=code_snippet,
        graph_section=graph_section,
    )


# ── Eval-mode prompts ─────────────────────────────────────────────────
# Used when the pipeline evaluates standalone code snippets from a
# labelled dataset rather than real SARIF findings.

EVAL_SYSTEM_PROMPT = """\
You are a senior application security engineer performing vulnerability triage.

A SAST scanner has flagged a code snippet as potentially vulnerable. Your job is \
to classify it as TRUE POSITIVE (vulnerable) or FALSE POSITIVE (not vulnerable).

Classification criteria — focus on whether DEFENSIVE MEASURES are present:

TRUE POSITIVE (vulnerable):
- Dangerous functions (eval, exec, os.system, pickle.loads, subprocess with \
shell=True, etc.) are called with user-controlled input and NO protective \
measures are applied.
- User input flows directly into a dangerous sink without any sanitization, \
validation, type checking, or restriction.
- Security-sensitive operations (password comparison, file access, etc.) are \
performed using insecure patterns with no mitigation.

FALSE POSITIVE (not vulnerable):
- The code uses a dangerous function BUT applies defensive measures: input \
validation, sanitization, escaping, allowlisting, type checking, AST \
restrictions, shell=False, parameterized queries, safe API alternatives, \
timing-safe comparisons, or any other form of protection.
- The code only operates on trusted/hardcoded/internal data — no user input \
reaches the dangerous operation.
- Even if the defensive measure is imperfect or theoretically bypassable, its \
PRESENCE means the developer has mitigated the risk. Classify as false positive.

Key principle: the question is "does the code include a protective measure \
against the vulnerability?" — NOT "could a skilled attacker bypass it?". \
If ANY defense exists, classify as false_positive. If NO defense exists, \
classify as true_positive.

Examples of defenses that make code a false positive:
- AST parsing/restriction before eval()/exec()
- subprocess.run() with shell=False or list-form commands
- Input validation (isalnum(), isinstance(), length checks, regex filters)
- Keyword/character blocklists or allowlists
- hmac.compare_digest() instead of == for secrets
- Escaping special characters before use
- Restricted builtins or sandboxed environments
- shlex.quote() for shell arguments

You may also be given CodeQL graph analysis with taint flow paths, sanitizers, \
call graphs, and sink/source classification. Use this data to support your \
classification.

Always respond with valid JSON. No markdown, no code fences — just the JSON object.\
"""

EVAL_USER_PROMPT_TEMPLATE = """\
## SAST Scanner Alert

A static analysis scanner flagged the following code as potentially vulnerable.

## Source Code

```python
{code_snippet}
```

{graph_section}

## Task

Analyze this code and determine whether the scanner's alert is correct. \
Is there a real, exploitable security vulnerability, or is this a false alarm?

Respond with this JSON structure:

{{
  "verdict": "true_positive" | "false_positive" | "uncertain",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<detailed explanation of your analysis>"
}}\
"""


def build_eval_prompt(
    code_snippet: str,
    codeql_context: str | None = None,
) -> str:
    """Build the user prompt for eval-mode code classification."""
    if codeql_context:
        graph_section = f"## CodeQL Graph Analysis\n\n```\n{codeql_context}\n```"
    else:
        graph_section = (
            "## CodeQL Graph Analysis\n\n"
            "_CodeQL analysis unavailable for this snippet. "
            "Base your analysis on the source code alone._"
        )

    return EVAL_USER_PROMPT_TEMPLATE.format(
        code_snippet=code_snippet,
        graph_section=graph_section,
    )
