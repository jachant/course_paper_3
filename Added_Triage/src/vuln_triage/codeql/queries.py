"""CodeQL query templates for vulnerability triage.

Each function returns a complete QL query string parameterised by the
file path and line number of a SARIF finding.  The queries target the
``codeql/python-all`` pack from the codeql-repo checkout.

Column names are stable contracts — :class:`CodeQLAnalyzer` parses
them by name.
"""

from __future__ import annotations


def _ql_escape(s: str) -> str:
    """Escape a string for safe inclusion inside a QL string literal."""
    return s.replace("\\", "/").replace('"', '\\"')


# =====================================================================
# 1. Taint Flow Paths
# =====================================================================

def taint_paths_simple_query(filepath: str, line: int) -> str:
    """Taint tracking from RemoteFlowSource to calls near the target line.

    Columns: source_label, source_file, source_line, sink_label, sink_file, sink_line
    """
    fp = _ql_escape(filepath)
    return f'''\
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking
import semmle.python.dataflow.new.RemoteFlowSources
import semmle.python.Concepts

module TargetTaintConfig implements DataFlow::ConfigSig {{
  predicate isSource(DataFlow::Node node) {{
    node instanceof RemoteFlowSource
  }}

  predicate isSink(DataFlow::Node node) {{
    node.getLocation().getFile().getRelativePath().matches("%{fp}")
    and node.getLocation().getStartLine() >= {line - 2}
    and node.getLocation().getStartLine() <= {line + 2}
  }}
}}

module TargetTaintFlow = TaintTracking::Global<TargetTaintConfig>;

from DataFlow::Node source, DataFlow::Node sink
where TargetTaintFlow::flow(source, sink)
select
  source.toString() as source_label,
  source.getLocation().getFile().getRelativePath() as source_file,
  source.getLocation().getStartLine() as source_line,
  sink.toString() as sink_label,
  sink.getLocation().getFile().getRelativePath() as sink_file,
  sink.getLocation().getStartLine() as sink_line
'''


# =====================================================================
# 2. Sanitizers & Guards
# =====================================================================

def sanitizers_query(filepath: str, line: int) -> str:
    """Identify sanitization / guarding calls near the finding.

    Columns: kind, name, file, line_number, code
    """
    fp = _ql_escape(filepath)
    return f'''\
import python

from Call call, Function enclosing
where
  call.getLocation().getFile().getRelativePath().matches("%{fp}")
  and enclosing.getLocation().getFile().getRelativePath().matches("%{fp}")
  and call.getLocation().getStartLine() >= enclosing.getLocation().getStartLine()
  and call.getLocation().getStartLine() <= enclosing.getLocation().getEndLine()
  and enclosing.getLocation().getStartLine() <= {line}
  and enclosing.getLocation().getEndLine() >= {line}
  and (
    call.getFunc().(Name).getId().regexpMatch(
      "(?i).*(escap|sanitiz|validat|quot|clean|strip|purg|encod|bleach|safe|protect|filter|check).*"
    )
    or
    call.getFunc().(Attribute).getName().regexpMatch(
      "(?i).*(escap|sanitiz|validat|quot|clean|strip|purg|encod|safe|protect|filter|check).*"
    )
    or
    call.getFunc().(Name).getId() = "isinstance"
  )
select
  "sanitizer" as kind,
  call.getFunc().toString() as name,
  call.getLocation().getFile().getRelativePath() as file,
  call.getLocation().getStartLine() as line_number,
  call.toString() as code
'''


# =====================================================================
# 3. Enclosing Scope
# =====================================================================

def scope_query(filepath: str, line: int) -> str:
    """Find the enclosing function/method and its decorators.

    Columns: kind, name, start_line, end_line, parameters, decorators
    """
    fp = _ql_escape(filepath)
    return f'''\
import python

from Function f
where
  f.getLocation().getFile().getRelativePath().matches("%{fp}")
  and f.getLocation().getStartLine() <= {line}
  and f.getLocation().getEndLine() >= {line}
  and not exists(Function inner |
    inner.getLocation().getFile().getRelativePath().matches("%{fp}")
    and inner.getLocation().getStartLine() > f.getLocation().getStartLine()
    and inner.getLocation().getEndLine() < f.getLocation().getEndLine()
    and inner.getLocation().getStartLine() <= {line}
    and inner.getLocation().getEndLine() >= {line}
  )
select
  f.getName() as name,
  f.getLocation().getStartLine() as start_line,
  f.getLocation().getEndLine() as end_line
'''


# =====================================================================
# 4. Call Graph Slice
# =====================================================================

def call_graph_query(filepath: str, line: int) -> str:
    """Callers of and callees from the enclosing function.

    Columns: direction, name, file, line_number
    """
    fp = _ql_escape(filepath)
    return f'''\
import python

from string direction, string name, string file, int line_number
where
  // Callees: calls made inside the enclosing function
  exists(Function target, Call c |
    target.getLocation().getFile().getRelativePath().matches("%{fp}")
    and target.getLocation().getStartLine() <= {line}
    and target.getLocation().getEndLine() >= {line}
    and not exists(Function inner |
      inner.getLocation().getFile().getRelativePath().matches("%{fp}")
      and inner.getLocation().getStartLine() > target.getLocation().getStartLine()
      and inner.getLocation().getEndLine() < target.getLocation().getEndLine()
      and inner.getLocation().getStartLine() <= {line}
      and inner.getLocation().getEndLine() >= {line}
    )
    and c.getScope() = target
    and direction = "callee"
    and name = c.getFunc().toString()
    and file = c.getLocation().getFile().getRelativePath()
    and line_number = c.getLocation().getStartLine()
  )
  or
  // Callers: calls to the enclosing function from elsewhere
  exists(Function target, Call c |
    target.getLocation().getFile().getRelativePath().matches("%{fp}")
    and target.getLocation().getStartLine() <= {line}
    and target.getLocation().getEndLine() >= {line}
    and not exists(Function inner |
      inner.getLocation().getFile().getRelativePath().matches("%{fp}")
      and inner.getLocation().getStartLine() > target.getLocation().getStartLine()
      and inner.getLocation().getEndLine() < target.getLocation().getEndLine()
      and inner.getLocation().getStartLine() <= {line}
      and inner.getLocation().getEndLine() >= {line}
    )
    and c.getFunc().(Name).getId() = target.getName()
    and c.getScope() != target
    and direction = "caller"
    and name = c.getScope().(Function).getName()
    and file = c.getLocation().getFile().getRelativePath()
    and line_number = c.getLocation().getStartLine()
  )
select direction, name, file, line_number
'''


# =====================================================================
# 5. Sink Classification
# =====================================================================

def sink_classification_query(filepath: str, line: int) -> str:
    """Classify the sink at/near the target line using stable CodeQL Concepts.

    Columns: sink_kind, api_name, file, line_number
    """
    fp = _ql_escape(filepath)
    return f'''\
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.Concepts

from DataFlow::Node node, string sink_kind, string api_name
where
  node.getLocation().getFile().getRelativePath().matches("%{fp}")
  and node.getLocation().getStartLine() >= {line - 1}
  and node.getLocation().getStartLine() <= {line + 1}
  and (
    exists(SqlExecution e |
      node = e.getSql() and
      sink_kind = "SqlExecution" and
      api_name = node.toString()
    )
    or
    exists(SystemCommandExecution e |
      node = e.getCommand() and
      sink_kind = "CommandExecution" and
      api_name = node.toString()
    )
    or
    exists(FileSystemAccess e |
      node = e.getAPathArgument() and
      sink_kind = "FileSystemAccess" and
      api_name = node.toString()
    )
    or
    exists(CodeExecution e |
      node = e.getCode() and
      sink_kind = "CodeExecution" and
      api_name = node.toString()
    )
  )
select
  sink_kind,
  api_name,
  node.getLocation().getFile().getRelativePath() as file,
  node.getLocation().getStartLine() as line_number
'''


# =====================================================================
# 6. Source Classification
# =====================================================================

def source_classification_query(filepath: str, line: int) -> str:
    """Classify flow sources that may reach the target line.

    Columns: source_kind, label, file, line_number
    """
    fp = _ql_escape(filepath)
    return f'''\
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking
import semmle.python.dataflow.new.RemoteFlowSources
import semmle.python.Concepts

module SourceFindConfig implements DataFlow::ConfigSig {{
  predicate isSource(DataFlow::Node node) {{
    node instanceof RemoteFlowSource
  }}

  predicate isSink(DataFlow::Node node) {{
    node.getLocation().getFile().getRelativePath().matches("%{fp}")
    and node.getLocation().getStartLine() >= {line - 2}
    and node.getLocation().getStartLine() <= {line + 2}
  }}
}}

module SourceFindFlow = TaintTracking::Global<SourceFindConfig>;

from DataFlow::Node source, DataFlow::Node sink
where SourceFindFlow::flow(source, sink)
select
  "RemoteFlowSource" as source_kind,
  source.toString() as label,
  source.getLocation().getFile().getRelativePath() as file,
  source.getLocation().getStartLine() as line_number
'''
