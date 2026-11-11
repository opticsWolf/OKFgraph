"""The agent skills shipped inside the wheel (0.13.1).

``okfgraph/skills/<name>/SKILL.md`` are harness-neutral sources: one MCP
finder (``okfgraph-mcp``), one shell finder (``okfgraph-cli``) and the
feeding discipline both share (``okfgraph-ingest``). ``okf setup skills``
copies them into a harness's skills folder — Claude Code reads
``.claude/skills/<name>/SKILL.md``, the default target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
DEFAULT_TARGET = Path(".claude") / "skills"


def bundled_skills() -> Dict[str, Path]:
    """Skill name -> its packaged ``SKILL.md``."""
    return {p.parent.name: p for p in sorted(SKILLS_DIR.glob("*/SKILL.md"))}


def setup_skills(target: Optional[Path] = None,
                 names: Optional[Iterable[str]] = None) -> dict:
    """Copy the packaged skills to ``target/<name>/SKILL.md``.

    ``names`` picks some (default: all three). A file already identical is
    left alone. Returns ``{"target", "installed", "updated", "unchanged"}``
    with skill names in each list.

    Raises:
        KeyError: a name that is not a packaged skill (the message lists them).
    """
    available = bundled_skills()
    chosen = list(names) if names else list(available)
    unknown = [n for n in chosen if n not in available]
    if unknown:
        raise KeyError(f"unknown skill(s) {unknown}; packaged: {sorted(available)}")
    root = Path(target) if target else DEFAULT_TARGET
    out: Dict[str, List[str]] = {"installed": [], "updated": [], "unchanged": []}
    for name in chosen:
        text = available[name].read_bytes()
        dest = root / name / "SKILL.md"
        if dest.is_file():
            bucket = "unchanged" if dest.read_bytes() == text else "updated"
        else:
            bucket = "installed"
        if bucket != "unchanged":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(text)
        out[bucket].append(name)
    return {"target": str(root), **out}
