"""Skills: different INSTRUCTIONS on demand, without hiring another agent. New in v9.

    from workflows.skills import SKILLS, load_skill
    load_skill.invoke({"skill_name": "deescalation"})     # the playbook's text

THE PATTERN
    When two specialists would differ only in what they are told, not in what
    they can do, they are not two agents: they are one agent with two
    playbooks. A skill is a playbook the model loads when it decides it needs
    it. Until then only its one-line description sits in the tool's docstring:
    the menu, not the cookbook (progressive disclosure). The benchmark's
    trade-off: the fewest calls of any pattern, but every loaded playbook stays
    in the context for the rest of the run, so tokens accumulate.

THE FILES ARE THE OPEN STANDARD'S
    skills/<name>/SKILL.md, YAML frontmatter (name, description) and a markdown
    body: the Agent Skills format (agentskills.io), the same one Claude Code
    loads. Session 15 writes real ones for Claude Code; these two are read by a
    LangChain tool instead. The tool's docstring is BUILT from the frontmatter,
    so adding a skill is adding a folder.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from langchain.tools import tool

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def read_skill(path: Path) -> dict:
    """One SKILL.md: frontmatter between the first two `---` lines, then the body."""
    _, front, body = path.read_text(encoding="utf-8").split("---", 2)
    meta = yaml.safe_load(front)
    return {"name": meta["name"], "description": meta["description"], "body": body.strip()}


SKILLS = {s["name"]: s for s in map(read_skill, sorted(SKILLS_DIR.glob("*/SKILL.md")))}


@tool
def load_skill(skill_name: str) -> str:
    """(replaced below, from the skills' own descriptions)"""
    skill = SKILLS.get(skill_name.strip())
    if skill is None:
        return f"No skill {skill_name!r}. Available: {', '.join(SKILLS)}."
    return skill["body"]


load_skill.description = (
    "Load a specialist playbook: instructions for one kind of situation, read before "
    "you reply. Available:\n"
    + "\n".join(f"- {s['name']}: {s['description']}" for s in SKILLS.values()))
