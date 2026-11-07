"""Read an MCP server's tools BEFORE you connect an agent to it. New in v9.

    from mcp_server.audit import audit
    for finding in audit(tools):          # tools from client.get_tools(), never called
        print(finding)

Connecting to an MCP server means trusting two things you did not write: its
tool DESCRIPTIONS, which the model reads on every call, and its tool OUTPUTS,
which arrive as data the model may follow as instructions. This checks the
first, by reading. It is a tripwire, not a guarantee: it finds the crude
attacks (hidden tags, "do not tell the user", arguments that invite data out),
and a careful attacker will write around it. The real defences are the rules in
session 12's MCP notebook, section 7: read-only by default, writes behind a person, pinned
and reviewed servers, and never all three legs of the lethal trifecta in one
agent.
"""
from __future__ import annotations

import re

# Phrases that address the MODEL rather than describe the tool.
RED_FLAGS = [
    (r"<\s*(important|system|instructions?)\s*>", "hidden instruction block"),
    (r"do not (mention|tell|inform|reveal)|don't (mention|tell)|without telling", "asks for secrecy"),
    (r"before (calling|using|answering)|always call|you must call|first call", "orders other calls"),
    (r"ignore (all |your |previous )", "overrides instructions"),
    (r"\b(email|password|api[_ ]?key|token|credit card|address)\b", "mentions personal data"),
    (r"https?://", "contains a URL"),
]
# Argument names that give data a way OUT of the conversation.
EXFIL_ARGS = {"notes", "context", "conversation", "history", "metadata", "debug", "extra"}


def audit(tools) -> list[str]:
    """One line per suspicious thing found in the tools' descriptions and arguments."""
    findings = []
    for t in tools:
        text = t.description or ""
        for pattern, what in RED_FLAGS:
            if re.search(pattern, text, re.IGNORECASE):
                findings.append(f"{t.name}: {what}")
        schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
        for arg in schema.get("properties", {}):
            if arg.lower() in EXFIL_ARGS:
                findings.append(f"{t.name}: argument `{arg}` could carry data out")
        if len(text) > 800:
            findings.append(f"{t.name}: a {len(text)}-character description; read all of it")
    return findings
