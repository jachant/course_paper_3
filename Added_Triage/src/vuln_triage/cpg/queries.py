"""CPGQL query templates for Joern CPG analysis.
Each template uses `__FILENAME__` and `__LINE__` placeholders
that are filled by `build_query()` via simple string replacement.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared preamble: find innermost method enclosing FILENAME:LINE
# ---------------------------------------------------------------------------
FIND_ENCLOSING = """
// List ALL methods in the file (materialised) for reliable filtering
val allMethods = cpg.method
  .filter(_.filename.endsWith("__FILENAME__"))
  .filterNot(_.fullName.endsWith(":<module>"))
  .l

// Filter: method must truly contain the target line
val enclosing = allMethods
  .filter(m => {
    val start = m.lineNumber.getOrElse(Int.MaxValue)
    val end = m.lineNumberEnd.getOrElse(-1)
    start > 0 && start <= __LINE__ && end >= __LINE__
  })
  .sortBy(m => m.lineNumberEnd.getOrElse(0) - m.lineNumber.getOrElse(0))

// Validate: ensure the candidate has AST nodes at/near the target line
val validated = if (enclosing.nonEmpty) {
  val best = enclosing.head
  val hasNodeAtLine = best.ast.lineNumber(__LINE__).nonEmpty ||
    best.ast.filter(n => n.lineNumber.exists(ln => ln >= __LINE__ - 1 && ln <= __LINE__ + 1)).nonEmpty
  if (hasNodeAtLine) enclosing
  else {
    // Fallback: find method containing a call at the target line
    val fallback = allMethods.filter(m =>
      m.ast.lineNumber(__LINE__).nonEmpty
    ).sortBy(m => m.lineNumberEnd.getOrElse(0) - m.lineNumber.getOrElse(0))
    if (fallback.nonEmpty) fallback else enclosing
  }
} else enclosing

val enclosing_final = validated
"""

# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------
METHOD_AT_LINE = """
{
  __FIND_ENCLOSING__
  if (enclosing_final.nonEmpty) {
    val m = enclosing_final.head
    // Collect code from AST nodes within method range (more reliable than m.code)
    val methodCode = m.ast
      .filter(n => {
        val ln = n.lineNumber.getOrElse(0)
        val start = m.lineNumber.getOrElse(0)
        val end = m.lineNumberEnd.getOrElse(0)
        ln >= start && ln <= end && ln > 0
      })
      .map(_.code)
      .l
      .distinct
      .filter(_.nonEmpty)
      .mkString("\\n")
      .take(500)  // Limit output size
    
    "METHOD: " + m.fullName + 
      " | lines " + m.lineNumber.getOrElse(0).toString + 
      "-" + m.lineNumberEnd.getOrElse(0).toString + 
      "\\n" + (if (methodCode.nonEmpty) methodCode else "<code extraction limited>")
  } else {
    "NO_ENCLOSING_METHOD_FOUND"
  }
}
"""

DATAFLOW_AT_LINE = """
{
  val sinks = cpg.call
    .filter(c => c.file.name.headOption.getOrElse("").endsWith("__FILENAME__")
      && c.lineNumber.exists(ln => ln >= __LINE__ - 2 && ln <= __LINE__ + 2))
    .filterNot(_.name.startsWith("<operator>"))
  
  // Filter sources to likely user-input parameters (heuristics)
  val sources = cpg.method
    .filter(_.filename.endsWith("__FILENAME__"))
    .filterNot(_.fullName.endsWith(":<module>"))
    .parameter
    .filter(p => {
      val name = p.name.toLowerCase
      name.contains("request") || name.contains("args") || 
      name.contains("input") || name.contains("data") ||
      name.contains("user") || name.contains("param") ||
      name.contains("query") || name.contains("form")
    })
  
  if (sinks.isEmpty) {
    "NO_SINKS_FOUND_AT_LINE"
  } else {
    try {
      val flows = sinks.reachableByFlows(sources).l
      if (flows.isEmpty) {
        // Fallback: show direct argument connections
        val directArgs = sinks.argument.code.l.distinct.take(5)
        val paramNames = sources.name.l.distinct.take(5)
        "NO_DATAFLOW_PATHS_FOUND\\n" +
        "Direct sink arguments: " + directArgs.mkString(", ") + "\\n" +
        "Potential sources: " + paramNames.mkString(", ")
      } else {
        flows.take(3)  // Reduced from 5 to limit output
          .map(path => path.elements
            .map(e => e.label + ": " + e.code.take(80)
              + " (line " + e.lineNumber.getOrElse(0).toString + ")")
            .mkString(" -> "))
          .mkString("\\n")
      }
    } catch {
      case e: Throwable => 
        // Fallback on error: show static argument info
        val sinkArgs = sinks.argument.code.l.distinct.take(5).mkString(", ")
        val sourceNames = sources.name.l.distinct.take(5).mkString(", ")
        "DATAFLOW_ERROR: " + e.getMessage.take(100) + 
        "\\nFallback — Sink args: " + sinkArgs + 
        "\\nPotential sources: " + sourceNames
    }
  }
}
"""

CALL_CONTEXT = """
{
  __FIND_ENCLOSING__
  if (enclosing_final.nonEmpty) {
    val target = enclosing_final.head
    val callerNames = target.start.caller.fullName.l.distinct
      .filterNot(_.contains(":<module>"))
      .take(10)  // Reduced limit
    val calleeNames = target.start.callee.fullName.l.distinct
      .filterNot(_.startsWith("<operator>"))
      .take(10)
    
    "CALLERS:\\n" + (if (callerNames.isEmpty) "(none)" else callerNames.mkString("\\n")) +
    "\\n\\nCALLEES:\\n" + (if (calleeNames.isEmpty) "(none)" else calleeNames.mkString("\\n"))
  } else {
    "CALLERS:\\n(no enclosing method found)\\n\\nCALLEES:\\n(no enclosing method found)"
  }
}
"""

METHOD_CALLS = """
{
  __FIND_ENCLOSING__
  if (enclosing_final.nonEmpty) {
    enclosing_final.head.start.call
      .filterNot(_.name.startsWith("<operator>"))
      .take(15)  // Limit output
      .map(c => c.name + "(" + 
        c.argument.code.l.mkString(", ").take(100) + ") " +
        "@ line " + c.lineNumber.getOrElse(0).toString)
      .l.distinct.mkString("\\n")
  } else {
    ""
  }
}
"""

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _escape_scala_string_content(s: str) -> str:
    """Escape a string for safe inclusion INSIDE Scala string quotes.

    Example: 'app.py' → 'app.py' (no change)
             'app"test.py' → 'app\\"test.py'
             'C:\\path\\file.py' → 'C:\\\\path\\\\file.py'
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def build_query(template: str, filename: str, line: int) -> str:
    """Fill a CPGQL template with concrete filename and line number.

    The template should contain "__FILENAME__" inside Scala string quotes,
    e.g.: _.filename.endsWith("__FILENAME__")
    This function escapes the filename content only (no outer quotes added).
    """
    # Normalize path separators and escape for Scala string content
    # Template already has quotes around __FILENAME__, so we don't add more
    safe_filename = _escape_scala_string_content(filename.replace("\\", "/"))

    result = template.replace("__FIND_ENCLOSING__", FIND_ENCLOSING)
    # ← БЕЗ дополнительных кавычек!
    result = result.replace("__FILENAME__", safe_filename)
    result = result.replace("__LINE__", str(line))
    return result
